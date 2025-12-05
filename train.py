import torch
import torch.nn as nn
import torch.optim as optim

def train_cgan(generator, discriminator, train_loader, device, epochs=20):
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    lambda_l1 = 100

    for epoch in range(1, epochs+1):
        generator.train()
        discriminator.train()

        for batch in train_loader:
            L = batch["L"].to(device)          # (B,1,H,W)
            ab = batch["ab"].to(device)        # (B,2,H,W)

            # ----------------------
            # Train Discriminator
            # ----------------------
            opt_D.zero_grad()

            fake_ab = generator(L)
            fake_pair = torch.cat([L, fake_ab.detach()], dim=1)
            real_pair = torch.cat([L, ab], dim=1)

            pred_fake = discriminator(fake_pair)
            pred_real = discriminator(real_pair)

            loss_D_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            loss_D_real = bce(pred_real, torch.ones_like(pred_real))
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            opt_D.step()

            # ----------------------
            # Train Generator
            # ----------------------
            opt_G.zero_grad()

            fake_ab = generator(L)
            fake_pair = torch.cat([L, fake_ab], dim=1)
            pred_fake = discriminator(fake_pair)

            adv_loss = bce(pred_fake, torch.ones_like(pred_fake))
            l1_loss = l1(fake_ab, ab)

            loss_G = adv_loss + lambda_l1 * l1_loss
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch}/{epochs} | G: {loss_G.item():.4f}, D: {loss_D.item():.4f}")
