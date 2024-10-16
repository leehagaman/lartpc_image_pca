
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time



class FeaturesToImage(nn.Module):
    def __init__(self, num_train, num_features):
        super(FeaturesToImage, self).__init__()
        
        self.embedding = nn.Embedding(num_train, num_features)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(16)
    
        self.deconv1 = nn.ConvTranspose2d(num_features, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is an array of indices of the training image, (batch_size)
        x = self.embedding(x) # (batch_size, num_features)
        x = x.view(x.size(0), x.size(1), 1, 1) # (batch_size, num_features, 1, 1)
        x = self.bn1(self.gelu(self.deconv1(x))) # (batch_size, 512, 2, 2)
        x = self.bn2(self.gelu(self.deconv2(x))) # (batch_size, 256, 4, 4)
        x = self.bn3(self.gelu(self.deconv3(x))) # (batch_size, 128, 8, 8)
        x = self.bn4(self.gelu(self.deconv4(x))) # (batch_size, 64, 16, 16)
        x = self.bn5(self.gelu(self.deconv5(x))) # (batch_size, 32, 32, 32)
        x = self.bn6(self.gelu(self.deconv6(x))) # (batch_size, 16, 64, 64)
        x = self.sigmoid(self.deconv7(x)) # (batch_size, 1, 128, 128)
        x = x.view(x.size(0), 128, 128)
        return x
    
    def get_image_from_features(self, x): # skip the embedding step, manually generating from features
        # x is a tensor of features, shape (batch_size, num_features)
        x = x.view(x.size(0), x.size(1), 1, 1) # (batch_size, num_features, 1, 1)
        x = self.bn1(self.gelu(self.deconv1(x))) # (batch_size, 512, 2, 2)
        x = self.bn2(self.gelu(self.deconv2(x))) # (batch_size, 256, 4, 4)
        x = self.bn3(self.gelu(self.deconv3(x))) # (batch_size, 128, 8, 8)
        x = self.bn4(self.gelu(self.deconv4(x))) # (batch_size, 64, 16, 16)
        x = self.bn5(self.gelu(self.deconv5(x))) # (batch_size, 32, 32, 32)
        x = self.bn6(self.gelu(self.deconv6(x))) # (batch_size, 16, 64, 64)
        x = self.sigmoid(self.deconv7(x)) # (batch_size, 1, 128, 128)
        x = x.view(x.size(0), 128, 128)
        return x

# modified MSE loss, to get the model to pay more attention to non-zero pixels
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    def forward(self, predicted, target):

        loss = (predicted - target) ** 2 # normal MSE loss

        target_large_mask = (target > 0.01)
        target_small_mask = (target <= 0.01)

        pred_large_mask = (predicted > 0.01)
        pred_small_mask = (predicted <= 0.01)

        overpred_mask = (predicted > target)
        underpred_mask = (predicted < target)

        large_overpred_mask = (pred_large_mask & target_small_mask & overpred_mask)
        large_underpred_mask = (pred_small_mask & target_large_mask & underpred_mask)

        loss_large_pixels = loss * target_large_mask
        loss_small_pixels = loss * target_small_mask

        large_overpred_loss = large_overpred_mask * loss
        large_underpred_loss = large_underpred_mask * loss

        #weighted_loss = loss_large_pixels + loss_small_pixels + large_overpred_loss * 10 + large_underpred_loss * 10000
        #weighted_loss = loss_large_pixels + loss_small_pixels + large_overpred_loss * 10

        weighted_loss = loss * (target + 0.001)

        return weighted_loss.sum()
    
