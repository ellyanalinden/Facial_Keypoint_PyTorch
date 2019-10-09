## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 11, stride = 4, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 192, out_channels = 384, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 *6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        
        x = self.relu(self.conv3(x))
        
        x = self.relu(self.conv4(x))
        
        x = self.relu(self.conv5(x))
        x = self.max_pool(x)
        
        x = x.view(x.size(0), 256 * 6 * 6)
        
        x = self.dropout(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
