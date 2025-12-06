import os
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Trains the conditional GAN (cGAN) for colorization using the provided generator and discriminator models.
def train_cgan(generator, discriminator, train_loader, val_loader, device, epochs=20, img_sample_idx=-1):
    # Move the models to the specified device (CPU or GPU)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Create optimizers for both the generator and the discriminator
    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Define the loss functions
    bce = nn.BCEWithLogitsLoss()        # Adversarial loss (real vs. fake)
    l1 = nn.L1Loss()                    # Reconstruction loss (accuracy in the generated color channels)
    mse = nn.MSELoss()                  # Mean Squared Error loss (for PSNR calculation)
    lambda_l1 = 100

    # Initialize the best validation loss for checkpointing
    best_val_loss = +float('inf')

    # Initialize lists to track various metrics across epochs
    g_losses = []
    d_losses = []
    val_L1_losses = []
    val_PSNR = []

    # ---- Training Loop ----
    for epoch in range(1, epochs+1):
        # Switch the models into training mode 
        generator.train()
        discriminator.train()

        # Initialize accumulators for the training metrics
        total_g = 0.0
        total_d = 0.0
        num_batches = 0

        # Load the training data in batches
        for batch in train_loader:
            # Separate the current batch's L/ab channels and move each to the specified device
            L = batch["L"].to(device)          # Shape: (batch_size, 1, 224, 224)
            ab = batch["ab"].to(device)        # Shape: (batch_size, 2, 224, 224)

            # ---- Train the Discriminator ----
            # Zero the gradients for the discriminator optimizer
            opt_D.zero_grad()

            # Generate fake ab channels, then create real and fake L/ab pairs
            fake_ab = generator(L)
            fake_pair = torch.cat([L, fake_ab.detach()], dim=1)
            real_pair = torch.cat([L, ab], dim=1)

            # Get the discriminator's predictions for both real and fake pairs
            pred_fake = discriminator(fake_pair)
            pred_real = discriminator(real_pair)

            # Compute the discriminator's loss (average of fake and real losses) and backpropagate
            loss_D_fake = bce(pred_fake, torch.zeros_like(pred_fake))   # D wants to output 0 for fake pairs
            loss_D_real = bce(pred_real, torch.ones_like(pred_real))    # D wants to output 1 for real pairs
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            opt_D.step()

            # ---- Train the Generator ----
            # Zero the gradients for the generator optimizer
            opt_G.zero_grad()

            # Generate fake ab channels, then create a fake L/ab pair to feed into the discriminator
            fake_ab = generator(L)
            fake_pair = torch.cat([L, fake_ab], dim=1)
            pred_fake = discriminator(fake_pair)

            # Compute the generator's adversarial and L1 losses, combine them (weighted sum), and backpropagate
            adv_loss = bce(pred_fake, torch.ones_like(pred_fake))       # G wants D to output 1 for fake pairs
            l1_loss = l1(fake_ab, ab)
            loss_G = adv_loss + lambda_l1 * l1_loss
            loss_G.backward()
            opt_G.step()

            # Update the running totals for the training metrics
            total_d += loss_D.item()
            total_g += loss_G.item()
            num_batches += 1

        # Compute the average generator and discriminator losses over the entire training set
        avg_g = total_g / num_batches
        avg_d = total_d / num_batches

        # ---- Validation Loop ----
        # Switch the generator into evaluation mode 
        generator.eval()

        # Initialize accumulators for the validation metrics 
        total_l1 = 0.0
        total_psnr = 0.0
        num_batches = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            # Load the validation data in batches
            for batch in val_loader:
                # Separate the current batch's L/ab channels and move each to the specified device
                L = batch["L"].to(device)
                ab = batch["ab"].to(device)

                # Generate fake ab channels
                fake_ab = generator(L)

                # Compute the L1 loss for this batch and add it to the running total
                l1_loss = l1(fake_ab, ab)
                total_l1 += l1_loss.item()

                # Compute the PSNR for this batch and add it to the running total
                mse_val = mse(fake_ab, ab).clamp(min=1e-8)
                psnr = 10 * torch.log10(4.0 / mse_val)
                total_psnr += psnr.item()

                # Increment the batch counter
                num_batches += 1

        # Compute the average L1 loss and PSNR over the entire validation set
        avg_l1 = total_l1 / num_batches
        avg_psnr = total_psnr / num_batches

        # Record the metrics for the current epoch
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        val_L1_losses.append(avg_l1)
        val_PSNR.append(avg_psnr)

        # Print the metrics for the current epoch
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"G: {avg_g:>7.3f} | "
            f"D: {avg_d:>7.3f} | "
            f"Val L1: {avg_l1:>7.4f} | "
            f"Val PSNR: {avg_psnr:>7.2f} dB"
        )

        # Checkpoint the model if the validation L1 loss has improved
        if avg_l1 < best_val_loss:
            best_val_loss = avg_l1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(generator.state_dict(), "checkpoints/best_generator.pth")
            torch.save(discriminator.state_dict(), "checkpoints/best_discriminator.pth")
            print(f"  -> Saved new best model at epoch {epoch} with validation L1 loss: {avg_l1:.4f}")

        # Save colorization progress photos every 5 epochs if img_sample_idx is specified
        if img_sample_idx >= 0 and (epoch == 1 or epoch % 5 == 0):
            # Load the L channel of the sample image from the validation dataset and add a batch dimension
            sample = val_loader.dataset[img_sample_idx]
            L = sample["L"].unsqueeze(0).to(device)

            # Disable gradient computation for evaluation
            with torch.no_grad():
                # Generate the predicted ab channels without the batch dimension
                ab_pred = generator(L)[0].cpu()

            # Convert to numpy for LAB-to-RGB conversion
            L_np       = sample["L"].cpu().squeeze().numpy()
            ab_pred_np = ab_pred.permute(1, 2, 0).numpy()

            # Convert from LAB to RGB color space
            rgb_pred = utils.lab_to_rgb(L_np, ab_pred_np)

            # Save the predicted image
            os.makedirs("progress_photos", exist_ok=True)
            plt.imsave(os.path.join("progress_photos", f"epoch_{epoch}.png"), rgb_pred)

    # Return the recorded metrics after training is complete
    return g_losses, d_losses, val_L1_losses, val_PSNR