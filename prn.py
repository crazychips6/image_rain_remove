import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU())
        self.res_conv1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv3 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv4 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv5 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
                                       nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, inputs):
        x = inputs
        for i in range(6):  # 迭代次数，不改变网络参数量
            x = torch.cat((inputs, x), 1)
            x = self.conv0(x)
            x = F.relu(self.res_conv1(x) + x)
            x = F.relu(self.res_conv2(x) + x)
            x = F.relu(self.res_conv3(x) + x)
            x = F.relu(self.res_conv4(x) + x)
            x = F.relu(self.res_conv5(x) + x)
            x = self.conv(x)
            x = x + inputs

        return x
