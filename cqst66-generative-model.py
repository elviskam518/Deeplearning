import os
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image

import cleanfid.fid as cfid
from torch.nn.utils import spectral_norm
from copy import deepcopy

cudnn.benchmark = True

nc = 3
nz = 128
ngf = 192
ndf = 32

batch_size = 128
beta1 = 0.5
max_steps = 50_000
lrD = 3e-4
lrG = 1.5e-4
ema_decay = 0.999


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if "Conv" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz=128, ngf=128, nc=3):
        super().__init__()
        self.fc = nn.Linear(nz, ngf * 4 * 4)
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, 4, 4)
        return self.main(x)

    def sample(self, z):
        return self.forward(z)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=32):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = spectral_norm(nn.Linear(ndf * 4 * 4 * 4, 1))

    def forward(self, x):
        h = self.main(x)
        h = h.view(h.size(0), -1)
        return self.head(h)


def d_hinge_loss(d_real, d_fake):
    loss_real = torch.relu(1.0 - d_real).mean()
    loss_fake = torch.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def g_hinge_loss(d_fake):
    return -d_fake.mean()
import torch
import torch.nn.functional as F

def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)

def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + x_mean

def rand_contrast(x):
    x_mean = x.mean(dim=(1,2,3), keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + x_mean
    

def DiffAugment(x, policy="color"):
    if policy == "" or policy is None:
        return x

    for p in policy.split(","):
        p = p.strip()
        if p == "color":
            x = rand_brightness(x)
            x = rand_saturation(x)
            x = rand_contrast(x)

    return x.clamp(-1, 1)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def setup_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


@torch.no_grad()
def ema_update(ema_model, model, decay=0.999):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if esd[k].dtype.is_floating_point:
            esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)
        else:
            esd[k].copy_(msd[k])


def main():
    manual_seed = 1
    print("Random Seed:", manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = dset.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset  = dset.CIFAR100(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True
    )

    print(f"> Size of training dataset: {len(train_dataset)}")
    print(f"> Size of test dataset: {len(test_dataset)}")

    train_iterator = iter(cycle(train_loader))

    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    netD = Discriminator(nc=nc, ndf=ndf).to(device)

    netG.apply(weights_init)
    netG_ema = deepcopy(netG).eval()
    for p in netG_ema.parameters():
        p.requires_grad_(False)

    n_params_G = sum(p.numel() for p in netG.parameters())
    n_params_D = sum(p.numel() for p in netD.parameters())
    n_params_total = n_params_G + n_params_D
    print(f"\n> Generator parameters: {n_params_G:,}")
    print(f"> Discriminator parameters: {n_params_D:,}")
    print(f"> Total parameters (G+D): {n_params_total:,}")
    print(" Parameter count is within limit (<1,000,000)" if n_params_total <= 1_000_000
          else "  WARNING: Total parameters exceed 1,000,000 limit!")
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.0, 0.9))

    print(f"\n> Starting training for {max_steps} steps...")

    steps = 0
    d_losses, g_losses = [], []

    while steps < max_steps:
        dloss_epoch, gloss_epoch = [], []

        for _ in range(1000):
            if steps >= max_steps:
                break

            real_data, _ = next(train_iterator)
            real_data = real_data.to(device)
            bs = real_data.size(0)

            netD.zero_grad(set_to_none=True)

            real_aug = DiffAugment(real_data, policy="color")

            z = torch.randn(bs, nz, device=device)
            fake = netG(z).detach()
            fake_aug = DiffAugment(fake, policy="color")
            d_real = netD(real_aug)
            d_fake = netD(fake_aug)

            errD = d_hinge_loss(d_real, d_fake)
            errD.backward()
            optimizerD.step()

            dloss_epoch.append(errD.item())

            netG.zero_grad(set_to_none=True)
            z = torch.randn(bs, nz, device=device)
            fake = netG(z)
            fake_aug_for_g = DiffAugment(fake, policy="color")
            d_fake_for_g = netD(fake_aug_for_g)

            errG = g_hinge_loss(d_fake_for_g)
            errG.backward()
            optimizerG.step()

            ema_update(netG_ema, netG, decay=ema_decay)

            gloss_epoch.append(errG.item())
            steps += 1

        avg_d = float(np.mean(dloss_epoch)) if len(dloss_epoch) else 0.0
        avg_g = float(np.mean(gloss_epoch)) if len(gloss_epoch) else 0.0
        d_losses.append(avg_d)
        g_losses.append(avg_g)

        print(f"Steps: {steps:5d}/{max_steps} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")
        if steps % 10000 == 0 or steps >= max_steps:
            netG_ema.eval()
            with torch.no_grad():
                sample_noise = torch.randn(64, nz, device=device)
                samples = netG_ema.sample(sample_noise)
                samples_vis = (samples * 0.5 + 0.5).clamp(0, 1)

            plt.figure(figsize=(8, 8))
            plt.imshow(vutils.make_grid(samples_vis, nrow=8).cpu().numpy().transpose(1, 2, 0))
            plt.title(f"Generated Samples (EMA) at Step {steps}")
            plt.axis("off")
            plt.savefig(f"samples_step_{steps}.png", dpi=150, bbox_inches="tight")
            plt.close()

    print(f"\n Training completed! Total G steps: {steps}")

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Iterations (x1000)")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss (Hinge)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n> Generating final batch of 64 samples (EMA)...")
    netG_ema.eval()
    with torch.no_grad():
        final_noise = torch.randn(64, nz, device=device)
        final_samples = netG_ema.sample(final_noise)
        final_samples_vis = (final_samples * 0.5 + 0.5).clamp(0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(vutils.make_grid(final_samples_vis, nrow=8).cpu().numpy().transpose(1, 2, 0))
    plt.title("Final Generated Samples (EMA, 64 images)")
    plt.axis("off")
    plt.savefig("final_64_samples.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n> Generating latent interpolations (EMA, 8 pairs)...")
    with torch.no_grad():
        n_interpolations = 8
        n_steps = 8
        all_interpolations = []

        for _ in range(n_interpolations):
            z0 = torch.randn(1, nz, device=device)
            z1 = torch.randn(1, nz, device=device)
            for t in np.linspace(0, 1, n_steps):
                z_interp = (1 - t) * z0 + t * z1
                all_interpolations.append(netG_ema.sample(z_interp))

        all_interpolations = torch.cat(all_interpolations, dim=0)
        interp_vis = (all_interpolations * 0.5 + 0.5).clamp(0, 1)

    plt.figure(figsize=(16, 16))
    plt.imshow(vutils.make_grid(interp_vis, nrow=n_steps).cpu().numpy().transpose(1, 2, 0))
    plt.title("Latent Space Interpolations (EMA, 8 pairs)")
    plt.axis("off")
    plt.savefig("interpolations.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n> Calculating FID scor...")

    real_images_dir = "real_images"
    generated_images_dir = "generated_images"
    num_samples = 10_000

    setup_directory(real_images_dir)
    setup_directory(generated_images_dir)

    print("  Saving 10,000 real test images...")
    for i in range(num_samples):
        img, _ = test_dataset[i]
        img = (img * 0.5 + 0.5).clamp(0, 1)
        save_image(img, os.path.join(real_images_dir, f"real_{i}.png"))
        if (i + 1) % 2000 == 0:
            print(f"    Saved: {i+1}/{num_samples}")

    print("  Generating 10,000 fake images ...")
    netG_ema.eval()
    num_generated = 0
    while num_generated < num_samples:
        with torch.no_grad():
            z = torch.randn(batch_size, nz, device=device)
            batch = netG_ema.sample(z).cpu()
            batch = (batch * 0.5 + 0.5).clamp(0, 1)

        for img in batch:
            if num_generated >= num_samples:
                break
            save_image(img, os.path.join(generated_images_dir, f"gen_{num_generated}.png"))
            num_generated += 1

        if num_generated % 2000 == 0:
            print(f"    Generated: {num_generated}/{num_samples}")

    print("  Computing FID score...")
    cfid.EXTENSIONS = {e.lower() for e in cfid.EXTENSIONS}
    fid_score = cfid.compute_fid(real_images_dir, generated_images_dir, mode="clean", num_workers=0)
    print(f"\n✅ FID Score : {fid_score:.2f}")

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Total Parameters (G+D): {n_params_total:,}")
    print("Parameter Limit: 1,000,000")
    print(f"Within Limit: {'✅ Yes' if n_params_total <= 1_000_000 else '❌ No'}")
    print()
    print(f"Total Training Steps (G updates): {steps:,}")
    print("Step Limit: 50,000")
    print(f"Within Limit: {'✅ Yes' if steps <= 50_000 else '❌ No'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
