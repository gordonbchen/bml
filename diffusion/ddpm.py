from argparse import ArgumentParser
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.datasets import MNIST

from einops import rearrange
from einops.layers.torch import Rearrange

from torchinfo import summary

import matplotlib.pyplot as plt
import wandb


torch.set_float32_matmul_precision("high")


def get_mnist() -> torch.Tensor:
    ds = MNIST(root="data", train=True, download=True)
    images = ds.data.to(dtype=torch.float32)
    images = (images / (255 / 2)) - 1
    assert (images.min() == -1) and (images.max() == 1)
    images = images.unsqueeze(-3)
    return images


def calc_cos_schedule(t: torch.Tensor, T: int, s: float = 0.008) -> torch.Tensor:
    phase = ((t / T) + s) / (1 + s)
    return torch.cos(phase * (torch.pi / 2.0)) ** 2.0


def calc_alpha_bar(t: torch.Tensor, T: int) -> torch.Tensor:
    return calc_cos_schedule(t, T) / calc_cos_schedule(torch.tensor(0.0, device=t.device), T)


def calc_alpha(alpha_bar: torch.Tensor) -> torch.Tensor:
    alpha = alpha_bar[1:] / alpha_bar[:-1]
    return alpha.clip(1e-4, 1 - 1e-4)


def add_noise(xs: torch.Tensor, t: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar = calc_alpha_bar(t, T)
    alpha_bar = rearrange(alpha_bar, "b -> b 1 1 1")
    noise = (1 - alpha_bar).sqrt() * torch.randn_like(xs)
    return noise, (alpha_bar.sqrt() * xs) + noise


class PosEnc(nn.Module):
    def __init__(self, max_pos: int, emb_dim: int, cycle_frac: float):
        super().__init__()
        self.register_buffer("pos_enc", torch.empty((max_pos, emb_dim), dtype=torch.float32))
        pos = torch.arange(max_pos)

        # Slowest sin/cos makes it frac of a full cycle.
        base = (max_pos / cycle_frac) / (2 * torch.pi)

        divs = base ** (2 * torch.arange(emb_dim // 2) / emb_dim)
        phase = pos[:, None] / divs
        self.pos_enc[pos, ::2] = phase.sin()
        self.pos_enc[pos, 1::2] = phase.cos()

    def forward(self, t: torch.Tensor):
        # t - 1 b/c t sampled from [1, T].
        return self.pos_enc[t - 1]


class ConvBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int):
        super().__init__()
        # TODO: weight standardized conv2d?
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, dim_out)

    def forward(self, x: torch.Tensor, scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None):
        x = self.norm(self.conv(x))
        if not (scale_shift is None):
            scale, shift = scale_shift
            x = ((1 + scale) * x) + shift  # TODO: where does scale shift come from?
        return F.gelu(x)


class ResBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, time_dim: int, groups: int, down: bool = False, up: bool = False):
        super().__init__()
        conv_dim = dim_out * 2 if up else dim_out
        self.conv1 = ConvBlock(dim_in, conv_dim, groups)
        self.conv2 = ConvBlock(conv_dim, conv_dim, groups)

        self.time_proj = nn.Sequential(nn.GELU(), nn.Linear(time_dim, conv_dim * 2))
        self.res_conv = nn.Identity() if dim_in == conv_dim else nn.Conv2d(dim_in, conv_dim, 1)

        assert not (down and up), "Cannot max pool and upsample."
        if down:
            self.post = nn.Sequential(
                Rearrange("b c (h x1) (w x2) -> b (c x1 x2) h w", x1=2, x2=2),
                nn.Conv2d(conv_dim * 4, dim_out, 3, padding=1)
            )
        elif up:
            self.post = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(conv_dim, dim_out, 3, padding=1)
            )
        else:
            self.post = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        time_emb = rearrange(self.time_proj(time_emb), "b c -> b c 1 1")
        z = self.conv1(x, time_emb.chunk(2, dim=1))
        z = self.conv2(z)
        z = z + self.res_conv(x)
        post = self.post(z)
        return (z, post)


class UNet(nn.Module):
    def __init__(self, img_dim: int, conv_dim: int, max_time: int, time_dim: int, groups: int, cycle_frac: float):
        super().__init__()
        self.down_conv1 = ResBlock(img_dim, conv_dim, time_dim, groups, down=True)
        self.down_conv2 = ResBlock(conv_dim, conv_dim * 2, time_dim, groups, down=True)

        self.bottleneck = ResBlock(conv_dim * 2, conv_dim * 4, time_dim, groups)

        self.up_conv1 = ResBlock(conv_dim * 4, conv_dim * 2, time_dim, groups, up=True)
        self.up_conv2 = ResBlock(conv_dim * 4, conv_dim, time_dim, groups, up=True)

        self.out_conv = ResBlock(conv_dim * 2, conv_dim * 2, time_dim, groups)
        self.to_pix = nn.Conv2d(conv_dim * 2, img_dim, 3, padding=1, bias=True)

        self.time_emb = nn.Sequential(
            PosEnc(max_time, time_dim, cycle_frac),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        time_emb = self.time_emb(t)
        res1, down1 = self.down_conv1(x, time_emb)
        res2, down2 = self.down_conv2(down1, time_emb)
        _, z = self.bottleneck(down2, time_emb)
        _, up2 = self.up_conv1(z, time_emb)
        _, up1 = self.up_conv2(torch.cat((up2, res2), dim=1), time_emb)
        _, pre_out = self.out_conv(torch.cat((up1, res1), dim=1), time_emb)
        out = self.to_pix(pre_out)
        return out
    

@torch.no_grad()
def sample_diffusion(model: nn.Module, batch_size: int, image_shape: tuple[int, int, int], max_time: int) -> torch.Tensor:
    model.eval()
    batched_shape = (batch_size, *image_shape)
    device = next(model.parameters()).device
    x = torch.randn(batched_shape, dtype=torch.float32, device=device)

    times = torch.arange(max_time + 1, device=device)
    alpha_bar = calc_alpha_bar(times, max_time)
    alpha = calc_alpha(alpha_bar)

    for t in range(max_time, 0, -1):
        time = times[t].expand(batch_size)
        pred_noise = ((1 - alpha[t-1]) / (1 - alpha_bar[t]).sqrt()) * model(x, time)
        x = (x - pred_noise) / alpha[t-1].sqrt()

        if t > 1:
            noise_scale = (1 - alpha[t-1]) * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t])
            # noise_scale = 1 - alpha[t-1]
            noise = torch.randn(batched_shape, dtype=torch.float32, device=device) * noise_scale.sqrt()
            x = x + noise
    model.train()
    return x


class CLIParams:
    def __post_init__(self) -> None:
        """Override params using cli args."""
        self.cli_override()

    def cli_override(self) -> None:
        """Override params from CLI args."""
        parser = ArgumentParser()
        for k, v in asdict(self).items():
            parser.add_argument(f"--{k}", type=type(v), default=v)
        args = parser.parse_args()

        for k, v in vars(args).items():
            setattr(self, k, v)

    def to_cli_args(self) -> list[str]:
        """Convert to params to CLI args."""
        args = []
        for k, v in asdict(self).items():
            args.append(f"--{k}={v}")
        return args


@dataclass
class Config(CLIParams):
    conv_dim: int = 32
    time_dim: int = 32
    groups: int = 4
    max_time: int = 128
    cycle_frac: float = 0.25

    # TODO: lr scheduling? https://www.desmos.com/calculator/1pi7ttmhhb
    lr: float = 3e-4

    batch_size: int = 128
    n_steps: int = 16_001

    name: str = ""


if __name__ == "__main__":
    config = Config()

    images = get_mnist()

    model = UNet(images.shape[1], config.conv_dim, config.max_time, config.time_dim, config.groups, config.cycle_frac)
    print(summary(model, input_data=(images[:config.batch_size], torch.ones(config.batch_size, dtype=torch.int64)), depth=1))
    optim = Adam(model.parameters(), lr=config.lr)

    model.to("cuda").compile()
    images = images.to("cuda")

    run = wandb.init(project="ddpm", config=asdict(config), name=(None if config.name == "" else config.name))
    for step in range(config.n_steps):
        xb = images[torch.randint(len(images), (config.batch_size,), device="cuda")]

        t = torch.randint(1, config.max_time + 1, (config.batch_size,), device="cuda")
        noise, xb_noised = add_noise(xb, t, config.max_time)

        pred_noise = model(xb_noised, t)

        loss = ((noise - pred_noise) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step % 500) == 0:
            print(f"{str(step):<5}: {loss.item()}")
            run.log({"loss": loss.item()}, step=step)
            
            if (step % 1000) == 0:
                samples = sample_diffusion(model, batch_size=16, image_shape=xb[0].shape, max_time=config.max_time).cpu()

                counts, bins = samples.histogram(bins=50)
                plt.plot((bins[1:] + bins[:-1]) / 2, counts)

                samples = rearrange(samples , "(row col) c h w -> c (row h) (col w)", row=4, col=4)
                samples = ((samples.clip(-1, 1) + 1) * (255 / 2)).to(dtype=torch.uint8)
                run.log({"samples": wandb.Image(samples), "sample_hist": plt}, step=step)

    run.finish()