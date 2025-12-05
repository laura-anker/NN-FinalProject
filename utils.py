import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Converts a LAB image to RGB
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

def show_prediction(generator, dataset, idx=0):
    generator.eval()
    sample = dataset[idx]

    L = sample["L"].unsqueeze(0).to(next(generator.parameters()).device)
    ab_true = sample["ab"]

    with torch.no_grad():
        ab_pred = generator(L).cpu()[0]

    # Convert to numpy for LAB-to-RGB
    L_np = L.cpu()[0].squeeze().numpy()
    ab_pred_np = ab_pred.permute(1,2,0).numpy()
    ab_true_np = ab_true.permute(1,2,0).numpy()

    rgb_pred = lab_to_rgb(L_np, ab_pred_np)
    rgb_true = lab_to_rgb(L_np, ab_true_np)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(L_np, cmap="gray"); plt.title("L"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(rgb_true); plt.title("True"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(rgb_pred); plt.title("Predicted"); plt.axis("off")
    plt.show()