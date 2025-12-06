from dataclasses import dataclass, asdict

import torch
torch.set_float32_matmul_precision("high")
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from einops import rearrange
from torchinfo import summary

import wandb
from tqdm import tqdm

import ddpm
import vae


@dataclass
class Config(ddpm.CLIParams):
    vae_path: str = "wandb/run-20251206_133624-uice77v0/files/model.pt"

    # DDPM.
    conv_dim: int = 16
    time_dim: int = 16
    down_blocks: int = 2
    groups: int = 4
    cycle_frac: float = 0.25
    ema_decay: float = 0.999

    max_time: int = 1000
    noise_schedule: str = "cos"

    dataset: str = "mnist"
    image_size: int = 32

    lr: float = 3e-4  # TODO: lr scheduling? https://www.desmos.com/calculator/1pi7ttmhhb
    batch_size: int = 128
    n_epochs: int = 32

    name: str = ""


if __name__ == "__main__":
    config = Config().cli_override()
    
    dataset = ddpm.get_dataset(config.dataset, config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    with open(config.vae_path.removesuffix("model.pt") + "/cli_args.txt", "r") as f:
        vae_cli_args = f.read()
    vae_config = vae.Config().cli_override(vae_cli_args.split())
    vae_model = vae.VAE(dataset[0][0].shape[0], vae_config.conv_dim, vae_config.groups, vae_config.down_convs, vae_config.latent_dim)
    vae_state_dict = torch.load(config.vae_path)
    vae_model.load_state_dict(vae_state_dict)
    vae_model.requires_grad_(False).eval()
    summary(vae_model, input_dim=(dataset[0][0].expand(config.batch_size, *dataset[0][0].shape)), depth=2)
    vae_latent_dummy = vae_model.enc(dataset[0][0].unsqueeze(0), sample=False)

    model = ddpm.UNet(vae_config.latent_dim, config.conv_dim, config.max_time, config.time_dim, config.down_blocks, config.groups, config.cycle_frac)
    summary(model, input_data=(vae_latent_dummy.expand(config.batch_size, *vae_latent_dummy.shape[1:]), torch.ones(config.batch_size, dtype=torch.int64)), depth=1)

    model.to("cuda").compile()
    vae_model.to("cuda").compile()

    ema_model = ddpm.EMAModel(model, decay=config.ema_decay)
    optim = Adam(model.parameters(), lr=config.lr)
    noise_schedule = ddpm.NOISE_SCHEDULES[config.noise_schedule](max_time=config.max_time).to("cuda")

    run = wandb.init(project="ldm", config=asdict(config), name=(None if config.name == "" else config.name))
    for epoch in range(config.n_epochs):
        for xb, _ in tqdm(dataloader):
            xb = xb.to("cuda")

            z = vae_model.enc(xb, sample=False)  # don't sample, just return the mean.

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
        sample_steps = torch.stack(sample_steps, dim=0)
        samples, sample_steps = vae_model.dec(samples), vae_model.dec(rearrange(sample_steps, "t b c h w -> (t b) c h w"))
        samples, sample_steps = samples.cpu(), sample_steps.cpu()
        samples = rearrange(ddpm.to_image(samples) , "(row col) c h w -> c (row h) (col w)", row=4, col=4)
        sample_steps = rearrange(ddpm.to_image(sample_steps) , "(t row col) c h w -> t c (row h) (col w)", t=32, row=4, col=4)
        if sample_steps.shape[1] == 1: sample_steps = sample_steps.expand(sample_steps.shape[0], 3, *sample_steps.shape[2:])
        video = wandb.Video(sample_steps.numpy(), format="gif", fps=8)
        run.log({"loss": loss.item(), "samples": wandb.Image(samples), "sample_steps": video}, step=epoch)

        torch.save(model.state_dict(), run.dir + f"/model.pt")
        torch.save(ema_model.model.state_dict(), run.dir + f"/ema_model.pt")
    run.finish()