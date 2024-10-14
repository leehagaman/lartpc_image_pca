
import torch
import torch.nn as nn


class FeaturesToImage(nn.Module):
    def __init__(self):
        super(FeaturesToImage, self).__init__()
        
        # 80 feature vectors, each is 100 dimensional
        self.embedding = nn.Embedding(80, 100)
    
        self.deconv1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.relu(self.deconv5(x))
        x = self.relu(self.deconv6(x))
        x = self.sigmoid(self.deconv7(x))
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
