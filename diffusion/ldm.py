from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from einops import rearrange
from einops.layers.torch import Rearrange
from torchinfo import summary

import wandb
from tqdm import tqdm

import ddpm
from cli_params import CLIParams


torch.set_float32_matmul_precision("high")


def conv_block(dim_in: int, dim_out: int, groups: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 3, padding=1, bias=False),
        nn.GroupNorm(groups, dim_out),
        nn.GELU(),
    )


def downsample(dim_in : int, dim_out: int) -> nn.Module:
    return nn.Sequential(
        Rearrange("b c (h x1) (w x2) -> b (c x1 x2) h w", x1=2, x2=2),
        nn.Conv2d(dim_in * 4, dim_out, 1)
    )


def upsample(dim_in : int, dim_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(dim_in, dim_out, 1)
    )


class Encoder(nn.Module):
    def __init__(self, img_dim: int, conv_dim: int, groups: int, down_convs: int, latent_dim: int):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(img_dim, conv_dim, 3, padding=1), nn.GELU())
        self.convs = nn.ModuleList()
        dim = conv_dim
        for _ in range(down_convs):
            self.convs.append(nn.Sequential(conv_block(dim, dim * 2, groups), downsample(dim*2, dim*2)))
            dim *= 2
        self.convs.append(conv_block(dim, dim, groups))
        self.mean_head = nn.Conv2d(dim, latent_dim, 3, padding=1)
        self.log_var_head = nn.Conv2d(dim, latent_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, sample: bool):
        x = self.in_conv(x)
        for conv in self.convs:
            x = conv(x)
        means = self.mean_head(x)
        if not sample:
            return means

        log_vars = self.log_var_head(x)
        stds = (0.5 * log_vars).exp()
        z = means + stds * torch.randn_like(means)
        return z, means, log_vars, stds
    

class Decoder(nn.Module):
    def __init__(self, img_dim: int, conv_dim: int, groups: int, down_convs: int, latent_dim: int):
        super().__init__()
        dim = conv_dim * (2 ** down_convs)
        self.convs = nn.ModuleList([conv_block(latent_dim, dim, groups)])
        for _ in range(down_convs):
            self.convs.append(nn.Sequential(conv_block(dim, dim, groups), upsample(dim, dim//2)))
            dim //= 2
        self.out_conv = nn.Sequential(
            conv_block(conv_dim, conv_dim, groups),
            conv_block(conv_dim, conv_dim, groups),
            nn.Conv2d(conv_dim, img_dim, 3, padding=1)
        )

    def forward(self, z: torch.Tensor):
        for conv in self.convs:
            z = conv(z)
        return self.out_conv(z)


class VAE(nn.Module):
    def __init__(self, img_dim: int, conv_dim: int, groups: int, down_convs: int, latent_dim: int):
        super().__init__()
        self.enc = Encoder(img_dim, conv_dim, groups, down_convs, latent_dim)
        self.dec = Decoder(img_dim, conv_dim, groups, down_convs, latent_dim)

    def forward(self, x: torch.Tensor):
        z, means, log_vars, stds = self.enc(x, sample=True)
        x_recon = self.dec(z)
        return x_recon, means, log_vars, stds 
    
    def calc_loss(self, x: torch.Tensor, kl_beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_recon, means, log_vars, stds = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        kl = 0.5 * ((means ** 2) + (stds ** 2) - log_vars - 1).sum(dim=1).mean()
        loss = recon_loss + kl_beta * kl
        return loss, recon_loss, kl, x_recon
    

@dataclass
class LDMConfig(CLIParams):
    # General.
    name: str = ""
    
    dataset: str = "mnist"
    image_size: int = 32

    # VAE.
    # TODO: nested CLIParams?
    vae_path: str = ""

    vae_epochs: int = 16
    vae_batch_size: int = 128
    vae_kl_beta: float = 0.1
    vae_lr: float = 3e-4

    vae_conv_dim: int = 32
    vae_groups: int = 8
    vae_down_convs: int = 2
    vae_latent_dim: int = 8

    # DDPM.
    conv_dim: int = 16
    time_dim: int = 16
    down_blocks: int = 2
    groups: int = 4
    cycle_frac: float = 0.25
    ema_decay: float = 0.999

    max_time: int = 1000
    noise_schedule: str = "cos"

    lr: float = 3e-4  # TODO: lr scheduling? https://www.desmos.com/calculator/1pi7ttmhhb
    batch_size: int = 128
    epochs: int = 32


def train_vae(vae: VAE, config: LDMConfig, dataloader: DataLoader, run: wandb.Run):
    vae.to("cuda").compile()
    optim = Adam(vae.parameters(), lr=config.lr)
    for epoch in range(config.vae_epochs):
        for x, _ in tqdm(dataloader):
            x = x.to("cuda")
            loss, recon_loss, kl, x_recon = vae.calc_loss(x, config.vae_kl_beta)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        x_recon = x_recon.detach()
        recon_images = torch.stack((x[:10], x_recon[:10]))
        recon_images = rearrange(ddpm.to_image(recon_images), "n b c h w -> c (n h) (b w)").cpu()
        run.log({"vae/loss": loss.item(), "vae/recon_loss": recon_loss.item(), "vae/kl": kl.item(), "vae/recons": wandb.Image(recon_images), "vae/epoch": epoch})
        print(f"{epoch}: loss={loss.item()}")
        torch.save(vae.state_dict(), run.dir + "/vae.pt")


if __name__ == "__main__":
    config = LDMConfig().cli_override()
    run = wandb.init(project="ldm", config=asdict(config), name=(None if config.name == "" else config.name))
    config.to_json_file(run.dir + "/cli_args.json")

    run.define_metric("vae/epoch")
    run.define_metric("vae/*", step_metric="vae/epoch")
    run.define_metric("ldm/epoch")
    run.define_metric("ldm/*", step_metric="ldm/epoch")
    
    dataset = ddpm.get_dataset(config.dataset, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    if config.vae_path == "":
        vae = VAE(dataset[0][0].shape[0], config.vae_conv_dim, config.vae_groups, config.vae_down_convs, config.vae_latent_dim)
        summary(vae, input_data=dataset[0][0].expand(config.vae_batch_size, *dataset[0][0].shape), depth=2)
        train_vae(vae, config, dataloader, run)
    else:
        vae_config = LDMConfig().from_json_file(Path(config.vae_path).parent / "cli_args.json")
        vae = VAE(dataset[0][0].shape[0], vae_config.vae_conv_dim, vae_config.vae_groups, vae_config.vae_down_convs, vae_config.vae_latent_dim)
        vae.load_state_dict(torch.load(config.vae_path))
        vae.to("cuda").requires_grad_(False).eval().compile()
        summary(vae, input_dim=(dataset[0][0].expand(config.batch_size, *dataset[0][0].shape)), depth=2)
    with torch.no_grad():
        vae_latent_dummy = vae.enc(dataset[0][0].unsqueeze(0).to("cuda"), sample=False).cpu()

    model = ddpm.UNet(config.vae_latent_dim, config.conv_dim, config.max_time, config.time_dim, config.down_blocks, config.groups, config.cycle_frac)
    summary(model, input_data=(vae_latent_dummy.expand(config.batch_size, *vae_latent_dummy.shape[1:]), torch.ones(config.batch_size, dtype=torch.int64)), depth=1)
    model.to("cuda").compile()

    ema_model = ddpm.EMAModel(model, decay=config.ema_decay)
    optim = Adam(model.parameters(), lr=config.lr)
    noise_schedule = ddpm.NOISE_SCHEDULES[config.noise_schedule](max_time=config.max_time).to("cuda")

    for epoch in range(config.epochs):
        for x, _ in tqdm(dataloader):
            with torch.no_grad():
                z = vae.enc(x.to("cuda"), sample=False)  # TODO: don't sample, just return the mean right?

            t = torch.randint(1, config.max_time + 1, (config.batch_size,), device="cuda")
            noise, z_noised = noise_schedule.add_noise(z, t)
            pred_noise = model(z_noised, t)

            loss = F.smooth_l1_loss(noise, pred_noise)  # TODO: test vs l1, l2.
            optim.zero_grad()
            loss.backward()
            optim.step()
            ema_model.update(model)

        print(f"{epoch}: {loss.item()}")
        samples, sample_steps = ddpm.sample_diffusion(
            ema_model.model, noise_schedule, batch_size=16, image_shape=z[0].shape,
            return_steps=torch.linspace(0, noise_schedule.max_time, steps=32, dtype=torch.int32)
        )
        with torch.no_grad():
            samples = vae.dec(samples)
            sample_steps = vae.dec(rearrange(sample_steps, "t b c h w -> (t b) c h w"))
        samples = rearrange(ddpm.to_image(samples) , "(row col) c h w -> c (row h) (col w)", row=4, col=4).cpu()
        sample_steps = rearrange(ddpm.to_image(sample_steps) , "(t row col) c h w -> t c (row h) (col w)", t=32, row=4, col=4).cpu()
        if sample_steps.shape[1] == 1: sample_steps = sample_steps.expand(sample_steps.shape[0], 3, *sample_steps.shape[2:])
        video = wandb.Video(sample_steps.numpy(), format="gif", fps=8)
        run.log({"ldm/loss": loss.item(), "ldm/samples": wandb.Image(samples), "ldm/sample_steps": video, "ldm/epoch": epoch})

        torch.save(model.state_dict(), run.dir + "/model.pt")
        torch.save(ema_model.model.state_dict(), run.dir + "/ema_model.pt")
    run.finish()