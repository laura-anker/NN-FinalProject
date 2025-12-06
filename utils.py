import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Converts from LAB to RGB color space
def lab_to_rgb(L_norm, ab_norm):
    # Undo the normalization
    L = L_norm * 255
    a = ab_norm[...,0] * 128.0 + 128.0
    b = ab_norm[...,1] * 128.0 + 128.0

    # Merge the three channels 
    lab = np.stack([L, a, b], axis=-1).astype("uint8")

    # Convert from LAB to RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb / 255, 0, 1)
    return rgb

# Displays an image based on the specified mode:
#    Mode 0: Both grayscale (L) and RGB (reconstructed from L/ab)
#    Mode 1: Grayscale only 
#    Mode 2: RGB (reconstructed from L/ab) only 
def show_image(image_data, mode=0, save_path=None):
    # Load the L channel and remove the batch dimension
    L_norm = image_data['L'].numpy()[0]                         # (1,224,224) -> (244,244)

    # If the mode requires RGB reconstruction, load the ab channels and convert
    if mode == 0 or mode == 2:
        ab_norm = image_data['ab'].numpy().transpose(1,2,0)     # (2,244,244) -> (224,224,2)
        rgb = lab_to_rgb(L_norm, ab_norm)

    # Visualize the image(s) based on the selected mode
    if mode == 0:
        # Save the grayscale and RGB images if a save path is provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.imsave(os.path.join(save_path, "grayscale.png"), L_norm, cmap="gray")
            plt.imsave(os.path.join(save_path, "rgb.png"), rgb)

        # Visualize the grayscale and RGB images side by side
        fig = plt.figure(figsize=(8, 4))
        gs = GridSpec(1, 2, figure=fig, left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.imshow(L_norm, cmap="gray")
        ax1.axis("off")
        ax2.imshow(rgb)
        ax2.axis("off")
        plt.show()

    elif mode == 1:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
        ax.imshow(L_norm, cmap="gray")
        ax.axis("off")
        plt.show()

    elif mode == 2:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
        ax.imshow(rgb)
        ax.axis("off")
        plt.show()

# Plots the generator and discriminator losses over epochs
def plot_gd_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label="Generator Loss", linewidth=2, color="#B77DD4")
    plt.plot(d_losses, label="Discriminator Loss", linewidth=2, color="#FFB2CC")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel("Epoch",)
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss Over Epochs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plots a validation metric (e.g., L1 loss or PSNR) over epochs
def plot_val_metric(metric_values, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(metric_values, linewidth=2, color="hotpink")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Validation {metric_name} Over Epochs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()