from dataclasses import dataclass, asdict

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


@dataclass
class Config(ddpm.CLIParams):
    epochs: int = 16
    batch_size: int = 128
    kl_beta: float = 0.1
    lr: float = 3e-4

    conv_dim: int = 32
    groups: int = 8
    latent_dim: int = 8
    down_convs: int = 2

    name: str = ""


if __name__ == "__main__":
    config = Config().cli_override()

    dataset = ddpm.get_dataset("mnist", image_size=32)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    vae = VAE(dataset[0][0].shape[0], config.conv_dim, config.groups, config.down_convs, config.latent_dim)
    summary(vae, input_data=dataset[0][0].expand(config.batch_size, *dataset[0][0].shape), depth=2)
    vae.to("cuda").compile()
    optim = Adam(vae.parameters(), lr=config.lr)

    run = wandb.init(project="vae", config=asdict(config), name=(None if config.name == "" else config.name))
    cli_args = config.to_cli_args()
    with open(run.dir + "/cli_args.txt", "w") as f:
        f.write(" ".join(cli_args))
    for epoch in range(config.epochs):
        for xb, _ in tqdm(dataloader):
            xb = xb.to("cuda")
            x_recon, means, log_vars, stds = vae(xb)

            recon_loss = F.mse_loss(x_recon, xb)
            kl = 0.5 * ((means ** 2) + (stds ** 2) - log_vars - 1).sum(dim=1).mean()
            loss = recon_loss + config.kl_beta * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        recon_images = torch.stack((xb[:10].to("cpu"), x_recon[:10].detach().to("cpu")))
        recon_images = ddpm.to_image(rearrange(recon_images, "n b c h w -> c (n h) (b w)"))
        run.log({"loss": loss.item(), "recon_loss": recon_loss.item(), "kl": kl.item(), "recons": wandb.Image(recon_images)}, step=epoch)
        print(f"{epoch:6d}: loss={loss.item():.4f}   recon_loss={recon_loss.item():.4f}   kl={kl.item():.4f}")
        torch.save(vae.state_dict(), run.dir + "/model.pt")
    run.finish()