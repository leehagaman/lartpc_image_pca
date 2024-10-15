
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 10
num_epochs = 100
lr = 0.001

y_train = torch.tensor(np.load("processed_data_files/nue_images.npz")["image_stack"]) / 100.
num_train = y_train.shape[0]
x_train = torch.tensor(np.arange(num_train))

# splitting into batches, not using the last batch which will have a different batch size
if y_train[-1].shape[0] != batch_size:
    print("skipping last batch, with different batch size")
    y_train = torch.split(y_train, batch_size)[:-1] 
    x_train = torch.split(x_train, batch_size)[:-1]
else:
    y_train = torch.split(y_train, batch_size)
    x_train = torch.split(x_train, batch_size)

num_features = 80

class FeaturesToImage(nn.Module):
    def __init__(self):
        super(FeaturesToImage, self).__init__()
        
        self.embedding = nn.Embedding(num_train, num_features)
    
        self.deconv1 = nn.ConvTranspose2d(num_features, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is an array of indices of the training image, (batch_size)
        x = self.embedding(x) # (batch_size, num_features)
        x = x.view(x.size(0), x.size(1), 1, 1) # (batch_size, num_features, 1, 1)
        x = self.relu(self.deconv1(x)) # (batch_size, 512, 2, 2)
        x = self.relu(self.deconv2(x)) # (batch_size, 256, 4, 4)
        x = self.relu(self.deconv3(x)) # (batch_size, 128, 8, 8)
        x = self.relu(self.deconv4(x)) # (batch_size, 64, 16, 16)
        x = self.relu(self.deconv5(x)) # (batch_size, 32, 32, 32)
        x = self.relu(self.deconv6(x)) # (batch_size, 16, 64, 64)
        x = self.sigmoid(self.deconv7(x)) # (batch_size, 1, 128, 128)
        x = x.view(x.size(0), 128, 128)
        return x
    
    def get_image_from_features(self, x): # skip the embedding step, manually generating from features
        # x is a tensor of features, shape (batch_size, num_features)
        x = x.view(x.size(0), x.size(1), 1, 1) # (batch_size, num_features, 1, 1)
        x = self.relu(self.deconv1(x)) # (batch_size, 512, 2, 2)
        x = self.relu(self.deconv2(x)) # (batch_size, 256, 4, 4)
        x = self.relu(self.deconv3(x)) # (batch_size, 128, 8, 8)
        x = self.relu(self.deconv4(x)) # (batch_size, 64, 16, 16)
        x = self.relu(self.deconv5(x)) # (batch_size, 32, 32, 32)
        x = self.relu(self.deconv6(x)) # (batch_size, 16, 64, 64)
        x = self.sigmoid(self.deconv7(x)) # (batch_size, 1, 128, 128)
        return x

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predicted, target):
        # Standard MSE loss (squared difference)
        loss = (predicted - target) ** 2
        
        # Weight the loss by the target values (higher weight for non-zero target pixels)
        weighted_loss = loss * (target + 0.001)
        
        # Normalize the loss by the sum of the weights (avoid division by zero)
        norm_factor = target.sum() + 1e-8  # Add small value to avoid division by zero
        weighted_loss = weighted_loss.sum() / norm_factor
        
        return weighted_loss

model = FeaturesToImage().to(device)
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

#criterion = nn.MSELoss()
criterion = WeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

os.system("rm -rf training_progress_images/*.png")

epoch_print_interval = 10

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
    print(f"Epoch {epoch} of {num_epochs}, Loss: {loss.item()}, Time: {epoch_time:.2f} seconds")
    if epoch % epoch_print_interval == 0:
        print(f"    Saving truth and generated image side by side...")
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(y_train_batch.detach().cpu().numpy()[0,:,:], cmap="jet")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(output.detach().cpu().numpy()[0,:,:], cmap="jet")
        plt.colorbar()
        plt.savefig(f"training_progress_images/epoch_{epoch}.png")
        plt.close()

