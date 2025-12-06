import os
import torch
import utils
import torch.nn as nn
import matplotlib.pyplot as plt

# Returns the average L1 loss and PSNR of the generator over the provided dataset
def evaluate_model(generator, data_loader, device):
    # Move the generator to the specified device (CPU or GPU)
    generator = generator.to(device)

    # Set the generator into evaluation mode
    generator.eval()

    # Define the loss functions for evaluation
    l1 = nn.L1Loss()                    # Reconstruction loss (accuracy in the generated color channels)
    mse = nn.MSELoss()                  # Mean Squared Error loss (for PSNR calculation)

    # Initialize accumulators for the evaluation metrics
    total_l1 = 0.0
    total_psnr = 0.0
    count = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Load the data in batches
        for batch in data_loader:
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
            count += 1

    # Compute the average L1 loss and PSNR over the entire dataset
    avg_l1 = total_l1 / count
    avg_psnr = total_psnr / count

    # Return the average L1 loss and PSNR
    return avg_l1, avg_psnr

# Predicts and visualizes the colorization of a single grayscale image using the trained generator model
def predict(generator, img_data, save_path=None):
    # Switch the generator into evaluation mode 
    generator.eval()

    # Load and prepare the L/ab channels 
    L = img_data["L"].unsqueeze(0).to(next(generator.parameters()).device)
    ab_true = img_data["ab"]

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Generate the predicted ab channels
        ab_pred = generator(L).cpu()[0]

    # Convert to numpy for LAB-to-RGB conversion
    L_np = L.cpu()[0].squeeze().numpy()
    ab_pred_np = ab_pred.permute(1,2,0).numpy()
    ab_true_np = ab_true.permute(1,2,0).numpy()

    # Convert from LAB to RGB color space
    rgb_pred = utils.lab_to_rgb(L_np, ab_pred_np)
    rgb_true = utils.lab_to_rgb(L_np, ab_true_np)

    # Save the predicted and true images if a save path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.imsave(os.path.join(save_path, "grayscale.png"), L_np, cmap="gray")
        plt.imsave(os.path.join(save_path, "true.png"), rgb_true)
        plt.imsave(os.path.join(save_path, "predicted.png"), rgb_pred)

    # Visualize the grayscale, true, and predicted images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.03, wspace=0.15)
    
    # Grayscale L channel
    axes[0].imshow(L_np, cmap="gray")
    axes[0].set_title("Grayscale (L)", fontsize=12)
    axes[0].axis("off")
    
    # True RGB (from LAB)
    axes[1].imshow(rgb_true)
    axes[1].set_title("True (L/ab)", fontsize=12)
    axes[1].axis("off")
    
    # Predicted RGB (from LAB)
    axes[2].imshow(rgb_pred)
    axes[2].set_title("Predicted (L/ab)", fontsize=12)
    axes[2].axis("off")

    plt.show()