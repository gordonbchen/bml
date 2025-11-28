from argparse import ArgumentParser
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.optim import Adam
from torchvision.datasets import MNIST

from einops import rearrange
from torchinfo import summary

import matplotlib.pyplot as plt
import wandb


def get_mnist() -> torch.Tensor:
    ds = MNIST(root="data", train=True, download=True)
    images = ds.data.to(dtype=torch.float32)
    images = (images / (255 / 2)) - 1
    assert (images.min() == -1) and (images.max() == 1)
    images = images.unsqueeze(-3)
    return images


def calc_cos_schedule(t: torch.Tensor, T: int, s: float = 0.1) -> torch.Tensor:
    phase = ((t / T) + s) / (1 + s)
    return torch.cos(phase * (torch.pi / 2.0)) ** 2.0


def calc_alpha_bar(t: torch.Tensor, T: int) -> torch.Tensor:
    return calc_cos_schedule(t, T) / calc_cos_schedule(torch.tensor(0.0, device=t.device), T)


def calc_alpha(alpha_bar: torch.Tensor) -> torch.Tensor:
    alpha = alpha_bar[1:] / alpha_bar[:-1]
    return alpha.maximum(torch.tensor(0.001, device=alpha_bar.device))


def add_noise(xs: torch.Tensor, t: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar = calc_alpha_bar(t, T)
    alpha_bar = rearrange(alpha_bar, "b -> b 1 1 1")
    noise = (1 - alpha_bar).sqrt() * torch.randn_like(xs)
    return noise, (alpha_bar.sqrt() * xs) + noise


class PosEnc(nn.Module):
    def __init__(self, max_pos: int, emb_dim: int):
        super().__init__()
        self.register_buffer("pos_enc", torch.empty((max_pos, emb_dim), dtype=torch.float32))
        pos = torch.arange(max_pos)

        # Slowest sin/cos makes it 0.25 of a full cycle.
        base = (max_pos / 0.25) / (2 * torch.pi)

        divs = base ** (2 * torch.arange(emb_dim // 2) / emb_dim)
        phase = pos[:, None] / divs
        self.pos_enc[pos, ::2] = phase.sin()
        self.pos_enc[pos, 1::2] = phase.cos()

    def forward(self, t: torch.Tensor):
        # t - 1 b/c t sampled from [1, T].
        return self.pos_enc[t - 1]


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        time_emb_dim: int, group_size: int,
        max_pool: bool = False, upsample: bool = False
    ):
        super().__init__()
        assert not (max_pool and upsample), "Cannot max pool and upsample."

        conv_out_channels = out_channels * 2 if upsample else out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(conv_out_channels // group_size, conv_out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(conv_out_channels // group_size, conv_out_channels),
            nn.ReLU()
        )

        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, conv_out_channels)
        )

        if in_channels == conv_out_channels:
            self.passthrough = nn.Identity()
        else:
            self.passthrough = nn.Conv2d(in_channels, conv_out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.post = nn.Identity()
        if max_pool:
            self.post = nn.MaxPool2d(2, 2)
        elif upsample:
            self.post = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(conv_out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(out_channels // group_size, out_channels),
                nn.ReLU()
            )

    def forward(self, xb: torch.Tensor, time_emb: torch.Tensor):
        z = self.conv1(xb)
        time_emb = rearrange(self.time_proj(time_emb), "b c -> b c 1 1")
        z = self.conv2(z + time_emb)
        z = z + self.passthrough(xb)
        post = self.post(z)
        return (z, post)


class UNet(nn.Module):
    def __init__(self, image_channels: int, conv_channels: int, max_time: int, time_emb_dim: int, group_size: int):
        super().__init__()
        self.down_conv1 = ConvBlock(image_channels, conv_channels, time_emb_dim, group_size, max_pool=True)
        self.down_conv2 = ConvBlock(conv_channels, conv_channels * 2, time_emb_dim, group_size, max_pool=True)

        self.bottleneck = ConvBlock(conv_channels * 2, conv_channels * 4, time_emb_dim, group_size)

        self.up_conv1 = ConvBlock(conv_channels * 4, conv_channels * 2, time_emb_dim, group_size, upsample=True)
        self.up_conv2 = ConvBlock(conv_channels * 4, conv_channels, time_emb_dim, group_size, upsample=True)

        self.out_conv = ConvBlock(conv_channels * 2, conv_channels * 2, time_emb_dim, group_size)
        self.to_pix = nn.Conv2d(conv_channels * 2, image_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.time_embedding = PosEnc(max_time, time_emb_dim)

    def forward(self, xb: torch.Tensor, t: torch.Tensor):
        time_emb = self.time_embedding(t)
        res1, down1 = self.down_conv1(xb, time_emb)
        res2, down2 = self.down_conv2(down1, time_emb)
        _, z = self.bottleneck(down2, time_emb)
        _, up2 = self.up_conv1(z, time_emb)
        _, up1 = self.up_conv2(torch.cat((up2, res2), dim=1), time_emb)
        _, pre_out = self.out_conv(torch.cat((up1, res1), dim=1), time_emb)
        out = self.to_pix(pre_out)
        return out
    

@torch.no_grad()
def sample_diffusion(model: nn.Module, batch_size: int, image_shape: tuple[int, int, int], max_diffusion_steps: int) -> torch.Tensor:
    model.eval()
    batched_shape = (batch_size, *image_shape)
    device = next(model.parameters()).device
    x = torch.randn(batched_shape, dtype=torch.float32, device=device)

    times = torch.arange(max_diffusion_steps + 1, device=device)
    alpha_bar = calc_alpha_bar(times, max_diffusion_steps)
    alpha = calc_alpha(alpha_bar)

    for t in range(max_diffusion_steps, 0, -1):
        time = times[t].expand(batch_size)
        pred_noise = ((1 - alpha[t-1]) / (1 - alpha_bar[t]).sqrt()) * model(x, time)
        x = (x - pred_noise) / alpha[t-1].sqrt()

        if t > 1:
            # noise_scale = (1 - alpha_bar[t-1]) * (1 - alpha[t-1]) / (1 - alpha_bar[t])
            noise_scale = 1 - alpha[t-1]
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


@dataclass
class Config(CLIParams):
    conv_channels: int = 32
    time_emb_dim: int = 32
    group_size: int = 4
    max_diffusion_steps: int = 1024

    lr: float = 1e-4
    batch_size: int = 64
    n_steps: int = 5_001


if __name__ == "__main__":
    config = Config()

    images = get_mnist()

    model = UNet(images.shape[1], config.conv_channels, config.max_diffusion_steps, config.time_emb_dim, config.group_size)
    print(summary(model, input_data=(images[:config.batch_size], torch.arange(config.batch_size))))
    optim = Adam(model.parameters(), lr=config.lr)

    model.to("cuda")
    images = images.to("cuda")

    run = wandb.init(project="ddpm", config=asdict(config))
    for step in range(config.n_steps):
        xb = images[torch.randint(len(images), (config.batch_size,), device="cuda")]

        t = torch.randint(1, config.max_diffusion_steps + 1, (config.batch_size,), device="cuda")
        noise, xb_noised = add_noise(xb, t, config.max_diffusion_steps)

        pred_noise = model(xb_noised, t)

        loss = ((noise - pred_noise) ** 2).sum() / config.batch_size
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step % 500) == 0:
            print(f"{str(step):<5}: {loss.item():.3f}")
            run.log({"loss": loss.item()}, step=step)
            
            if (step % 1000) == 0:
                samples = sample_diffusion(model, batch_size=16, image_shape=xb[0].shape, max_diffusion_steps=config.max_diffusion_steps).cpu()

                density, bins = samples.histogram(bins=50)
                plt.plot((bins[1:] + bins[:-1]) / 2, density)

                samples = rearrange(samples , "(row col) c h w -> c (row h) (col w)", row=4, col=4)
                samples = ((samples.clip(-1, 1) + 1) * (255 / 2)).to(dtype=torch.uint8)
                run.log({"samples": wandb.Image(samples), "sample_hist": plt}, step=step)

    run.finish()