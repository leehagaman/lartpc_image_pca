
import torch
import torch.nn as nn
import numpy as np

y_train = np.load("processed_data_files/nue_images.npz")["image_stack"]
num_train = y_train.shape[0]

batch_size = 100
num_features = 80

class FeaturesToImage(nn.Module):
    def __init__(self):
        super(FeaturesToImage, self).__init__()
        
        self.embedding = nn.Embedding(num_train, num_features)
    
        self.deconv1 = nn.ConvTranspose2d(80, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is an int, the index of the training image
        x = self.embedding(x) # (batch_size, num_features)
        x = x.view(x.size(0), x.size(1), 1, 1) # (batch_size, num_features, 1, 1)
        x = self.relu(self.deconv1(x)) # (batch_size, 512, 2, 2)
        x = self.relu(self.deconv2(x)) # (batch_size, 256, 4, 4)
        x = self.relu(self.deconv3(x)) # (batch_size, 128, 8, 8)
        x = self.relu(self.deconv4(x)) # (batch_size, 64, 16, 16)
        x = self.relu(self.deconv5(x)) # (batch_size, 32, 32, 32)
        x = self.relu(self.deconv6(x)) # (batch_size, 16, 64, 64)
        x = self.sigmoid(self.deconv7(x)) # (batch_size, 1, 128, 128)
        return x
    
    def get_image_from_features(self, features):
        x = features
        x = x.view(x.size(0), x.size(1), 1, 1) # (batch_size, num_features, 1, 1)
        x = self.relu(self.deconv1(x)) # (batch_size, 512, 2, 2)
        x = self.relu(self.deconv2(x)) # (batch_size, 256, 4, 4)
        x = self.relu(self.deconv3(x)) # (batch_size, 128, 8, 8)
        x = self.relu(self.deconv4(x)) # (batch_size, 64, 16, 16)
        x = self.relu(self.deconv5(x)) # (batch_size, 32, 32, 32)
        x = self.relu(self.deconv6(x)) # (batch_size, 16, 64, 64)
        x = self.sigmoid(self.deconv7(x)) # (batch_size, 1, 128, 128)
        return x

lr = 0.001

model = FeaturesToImage()
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dummy_input = torch.zeros(100, dtype=torch.long)
output = model(dummy_input)
print(f"Output shape: {output.shape}")
assert output.shape[1:] == (1, 128, 128), "Output shape mismatch"

dummy_features = torch.zeros(1, num_features)
output = model.get_image_from_features(dummy_features)
print(f"Output shape: {output.shape}")
assert output.shape[1:] == (1, 128, 128), "Output shape mismatch"
