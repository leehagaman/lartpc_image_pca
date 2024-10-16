import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from model import FeaturesToImage, WeightedMSELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 100
num_epochs = 100
lr = 0.001
num_features = 80

y_train = torch.tensor(np.load("processed_data_files/nue_images_combined.npz")["image_stack"]) / 100.
num_train = y_train.shape[0]
print(f"loaded {num_train} training images")
x_train = torch.tensor(np.arange(num_train))

# splitting into batches, not using the last batch which will have a different batch size
if y_train[-1].shape[0] != batch_size:
    print("skipping last batch, with different batch size")
    y_train = torch.split(y_train, batch_size)[:-1] 
    x_train = torch.split(x_train, batch_size)[:-1]
else:
    y_train = torch.split(y_train, batch_size)
    x_train = torch.split(x_train, batch_size)


model = FeaturesToImage(num_train, num_features).to(device)

print("Compiling model...", end="")
model = torch.compile(model)
print("done")

num_embedding_params = sum(p.numel() for p in model.embedding.parameters())
num_total_params = sum(p.numel() for p in model.parameters())
num_inference_params = num_total_params - num_embedding_params

print(f"Number of embedding parameters (training only): {num_embedding_params}")
print(f"Number of inference parameters: {num_inference_params}")

criterion = WeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

os.system("rm -rf training_progress_images/*.png")

epoch_print_interval = 20

# training loop
print("Starting training loop...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    for batch_num in range(len(x_train)):
        x_train_batch = x_train[batch_num].to(device)
        y_train_batch = y_train[batch_num].to(device)
        optimizer.zero_grad()
        output = model(x_train_batch)
        loss = criterion(output, y_train_batch)
        loss.backward()
        optimizer.step()
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    if epoch % epoch_print_interval == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch} of {num_epochs}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")
        print(f"    Saving truth and generated images")
        fig, axs = plt.subplots(2, 5, figsize=(10, 5))
        for event_i in range(5):
            axs[0, event_i].imshow(y_train_batch.detach().cpu().numpy()[event_i,:,:], cmap="jet", vmin=0, vmax=1)
            axs[1, event_i].imshow(output.detach().cpu().numpy()[event_i,:,:], cmap="jet", vmin=0, vmax=1)
        for ax in axs.flatten():
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"training_progress_images/epoch_{epoch:04d}.png")
        plt.close()

print("Training done!")
