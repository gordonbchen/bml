from copy import deepcopy
from argparse import ArgumentParser
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, OxfordIIITPet
from torchvision.transforms import v2

from einops import rearrange
from einops.layers.torch import Rearrange
from torchinfo import summary

import matplotlib.pyplot as plt
import wandb


torch.set_float32_matmul_precision("high")


class ImagesOnly:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        return img


def get_dataset(name: str, image_size: int) -> Dataset:
    DATASETS = {"mnist": (MNIST, {}), "pets": (OxfordIIITPet, {"target_types": "binary-category"})}
    klass, kwargs = DATASETS[name]
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        lambda x: (x * 2) - 1,  # [0, 1] to [-1, 1].
    ])
    return ImagesOnly(klass(root="data", transform=transform, download=True, **kwargs))


def inf_dataloader(dataloader: DataLoader):
    while True:
        for xb in dataloader:
            yield xb


class NoiseSchedule(nn.Module):
    def __init__(self, max_time: int):
        super().__init__()
        self.max_time = max_time

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = rearrange(self.alpha_bar[t], "b -> b 1 1 1")
        noise = (1 - alpha_bar).sqrt() * torch.randn_like(x)
        return noise, (alpha_bar.sqrt() * x) + noise


class CosNoiseSchedule(NoiseSchedule):
    def __init__(self, max_time: int, s: float = 0.008, alpha_clip: float = 1e-4):
        super().__init__(max_time)
        t = torch.arange(-1, max_time + 1, dtype=torch.float32)
        phase = ((t / max_time) + s) / (1 + s)
        cos2 = torch.cos(phase * (torch.pi / 2.0)) ** 2.0
        alpha_bar = cos2 / cos2[0]

        alpha = (alpha_bar[1:] / alpha_bar[:-1]).clip(alpha_clip, 1 - alpha_clip)
        beta = 1 - alpha
        alpha_bar = alpha_bar[1:]
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)


class LinearNoiseSchedule(NoiseSchedule):
    def __init__(self, max_time: int, beta_min: float = 1e-4, beta_max: float = 0.02):
        super().__init__(max_time)
        scale = 1000 / max_time
        beta = torch.linspace(scale * beta_min, scale * beta_max, max_time + 1, dtype=torch.float32)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)


NOISE_SCHEDULES: dict[str, type[NoiseSchedule]] = {
    "cos": CosNoiseSchedule,
    "linear": LinearNoiseSchedule,
}


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
    def __init__(self, img_dim: int, conv_dim: int, max_time: int, time_dim: int, down_convs: int, groups: int, cycle_frac: float):
        super().__init__()
        self.init_conv = nn.Conv2d(img_dim, conv_dim, 3, padding=1)

        self.down_convs = nn.ModuleList()
        in_dim = conv_dim
        for _ in range(down_convs):
            self.down_convs.append(ResBlock(in_dim, in_dim * 2, time_dim, groups, down=True))
            in_dim *= 2

        # TODO: add attention.
        self.bottleneck = ResBlock(in_dim, in_dim * 2, time_dim, groups)
        in_dim *= 2

        self.up_convs = nn.ModuleList([ResBlock(in_dim, in_dim // 2, time_dim, groups, up=True)])
        in_dim //= 2

        for _ in range(down_convs - 1):
            self.up_convs.append(ResBlock(in_dim * 2, in_dim // 2, time_dim, groups, up=True))
            in_dim //= 2
        self.up_convs.append(ResBlock(in_dim * 2, in_dim // 2, time_dim, groups))

        self.to_pix = nn.Conv2d(conv_dim, img_dim, 3, padding=1)

        self.time_emb = nn.Sequential(
            PosEnc(max_time, time_dim, cycle_frac),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        time_emb = self.time_emb(t)
        x = self.init_conv(x)

        down_res = []
        for conv in self.down_convs:
            res, x = conv(x, time_emb)
            down_res.append(res)

        _, x = self.bottleneck(x, time_emb)

        _, x = self.up_convs[0](x, time_emb)
        for (conv, res) in zip(self.up_convs[1:], reversed(down_res)):
            _, x = conv(torch.cat((x, res), dim=1), time_emb)

        x = self.to_pix(x)
        return x
    

@torch.no_grad()
def sample_diffusion(model: nn.Module, noise_schedule: NoiseSchedule, batch_size: int, image_shape: tuple[int, int, int]) -> torch.Tensor:
    batched_shape = (batch_size, *image_shape)
    device = next(model.parameters()).device
    x = torch.randn(batched_shape, dtype=torch.float32, device=device)

    for t in range(noise_schedule.max_time, 0, -1):
        time = torch.tensor([t], dtype=torch.int64, device=device).expand(batch_size)
        pred_noise = ((1 - noise_schedule.alpha[t]) / (1 - noise_schedule.alpha_bar[t]).sqrt()) * model(x, time)
        x = (x - pred_noise) / noise_schedule.alpha[t].sqrt()

        if t > 1:
            noise = torch.randn(batched_shape, dtype=torch.float32, device=device) * noise_schedule.beta[t].sqrt()
            x = x + noise
    return x


class EMAModel:
    def __init__(self, model: nn.Module, decay: float):
        self.model = deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        decay = min(self.decay, 1 - (1 / (step + 1)))
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            v.mul_(decay).add_(msd[k], alpha=1 - decay)


class CLIParams:
    def cli_override(self):
        """Override params from CLI args."""
        parser = ArgumentParser()
        for k, v in asdict(self).items():
            parser.add_argument(f"--{k}", type=type(v), default=v)
        args = parser.parse_args()

        for k, v in vars(args).items():
            setattr(self, k, v)
        return self

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
    down_blocks: int = 3
    groups: int = 4
    cycle_frac: float = 0.25
    ema_decay: float = 0.999

    max_time: int = 1000
    noise_schedule: str = "cos"

    dataset: str = "mnist"
    image_size: int = 32

    lr: float = 3e-4  # TODO: lr scheduling? https://www.desmos.com/calculator/1pi7ttmhhb
    batch_size: int = 128
    n_steps: int = 16_001
    log_steps: int = 1_000

    name: str = ""


if __name__ == "__main__":
    config = Config().cli_override()

    dataset = get_dataset(config.dataset, config.image_size)
    dataloader = inf_dataloader(DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True))

    model = UNet(dataset[0].shape[0], config.conv_dim, config.max_time, config.time_dim, config.down_blocks, config.groups, config.cycle_frac)
    summary(model, input_data=(dataset[0].unsqueeze(0).repeat(config.batch_size, 1, 1, 1), torch.ones(config.batch_size, dtype=torch.int64)), depth=1)

    model.to("cuda").train().compile()
    ema_model = EMAModel(model, decay=config.ema_decay)
    optim = Adam(model.parameters(), lr=config.lr)

    noise_schedule = NOISE_SCHEDULES[config.noise_schedule](max_time=config.max_time)
    noise_schedule.to("cuda")

    run = wandb.init(project="ddpm", config=asdict(config), name=(None if config.name == "" else config.name))
    for step in range(config.n_steps):
        xb = next(dataloader).to("cuda")

        t = torch.randint(1, config.max_time + 1, (config.batch_size,), device="cuda")
        noise, xb_noised = noise_schedule.add_noise(xb, t)

        pred_noise = model(xb_noised, t)

        loss = F.smooth_l1_loss(noise, pred_noise)
        optim.zero_grad()
        loss.backward()
        optim.step()

        ema_model.update(model, step)

        if (step % config.log_steps) == 0:
            print(f"{str(step):<5}: {loss.item()}")
            samples = sample_diffusion(ema_model.model, noise_schedule, batch_size=16, image_shape=xb[0].shape).cpu()

            counts, bins = samples.histogram(bins=50)
            plt.plot((bins[1:] + bins[:-1]) / 2, counts)

            samples = rearrange(samples , "(row col) c h w -> c (row h) (col w)", row=4, col=4)
            samples = ((samples.clip(-1, 1) + 1) * (255 / 2)).to(dtype=torch.uint8)
            run.log({"loss": loss.item(), "samples": wandb.Image(samples), "sample_hist": plt}, step=step)
    run.finish()