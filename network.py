import torch
import torch.nn as nn
from torch.nn import  functional as F
import numpy as np
import os
    
class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,3,stride, 1,bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
                nn.BatchNorm2d(outchannel) )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

    def save(self, dir, conv_num, bn_num):
        conv1 = self.left[0].weight.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "conv{}_weight.npy".format(str(conv_num))), conv1)
        conv_num += 1

        bn1w = self.left[1].weight.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_weight.npy".format(str(bn_num))), bn1w)
        bn1b = self.left[1].bias.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_bias.npy".format(str(bn_num))), bn1b)
        bn1m = self.left[1].running_mean.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_mean.npy".format(str(bn_num))), bn1m)
        bn1v = self.left[1].running_var.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_var.npy".format(str(bn_num))), bn1v)
        bn_num += 1

        conv2 = self.left[3].weight.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "conv{}_weight.npy".format(str(conv_num))), conv2)
        conv_num += 1

        bn2w = self.left[4].weight.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_weight.npy".format(str(bn_num))), bn2w)
        bn2b = self.left[4].bias.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_bias.npy".format(str(bn_num))), bn2b)
        bn2m = self.left[4].running_mean.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_mean.npy".format(str(bn_num))), bn2m)
        bn2v = self.left[4].running_var.data.detach().cpu().numpy()
        np.save(os.path.join(dir, "bn{}_var.npy".format(str(bn_num))), bn2v)
        bn_num += 1

        if self.right is not None:
            conv3 = self.right[0].weight.data.detach().cpu().numpy()
            np.save(os.path.join(dir, "conv{}_weight.npy".format(str(conv_num))), conv3)
            conv_num += 1

            bn3w = self.right[1].weight.data.detach().cpu().numpy()
            np.save(os.path.join(dir, "bn{}_weight.npy".format(str(bn_num))), bn3w)
            bn3b = self.right[1].bias.data.detach().cpu().numpy()
            np.save(os.path.join(dir, "bn{}_bias.npy".format(str(bn_num))), bn3b)
            bn3m = self.right[1].running_mean.data.detach().cpu().numpy()
            np.save(os.path.join(dir, "bn{}_mean.npy".format(str(bn_num))), bn3m)
            bn3v = self.right[1].running_var.data.detach().cpu().numpy()
            np.save(os.path.join(dir, "bn{}_var.npy".format(str(bn_num))), bn3v)
            bn_num += 1

        return conv_num, bn_num

    def load(self, dir, conv_num, bn_num):
        conv1 = np.load(os.path.join(dir, "conv{}_weight.npy".format(conv_num)))
        self.left[0].weight.data = torch.from_numpy(conv1)
        conv_num += 1

        bn1w = np.load(os.path.join(dir, "bn{}_weight.npy".format(bn_num)))
        self.left[1].weight.data = torch.from_numpy(bn1w)
        bn1b = np.load(os.path.join(dir, "bn{}_bias.npy".format(bn_num)))
        self.left[1].bias.data = torch.from_numpy(bn1b)
        bn1m = np.load(os.path.join(dir, "bn{}_mean.npy".format(bn_num)))
        self.left[1].running_mean.data = torch.from_numpy(bn1m)
        bn1v = np.load(os.path.join(dir, "bn{}_var.npy".format(bn_num)))
        self.left[1].running_var.data = torch.from_numpy(bn1v)
        bn_num += 1

        conv2 = np.load(os.path.join(dir, "conv{}_weight.npy".format(conv_num)))
        self.left[3].weight.data = torch.from_numpy(conv2)
        conv_num += 1

        bn2w = np.load(os.path.join(dir, "bn{}_weight.npy".format(bn_num)))
        self.left[4].weight.data = torch.from_numpy(bn2w)
        bn2b = np.load(os.path.join(dir, "bn{}_bias.npy".format(bn_num)))
        self.left[4].bias.data = torch.from_numpy(bn2b)
        bn2m = np.load(os.path.join(dir, "bn{}_mean.npy".format(bn_num)))
        self.left[4].running_mean.data = torch.from_numpy(bn2m)
        bn2v = np.load(os.path.join(dir, "bn{}_var.npy".format(bn_num)))
        self.left[4].running_var.data = torch.from_numpy(bn2v)
        bn_num += 1

        if self.right is not None:
            conv3 = np.load(os.path.join(dir, "conv{}_weight.npy".format(conv_num)))
            self.right[0].weight.data = torch.from_numpy(conv3)
            conv_num += 1

            bn3w = np.load(os.path.join(dir, "bn{}_weight.npy".format(bn_num)))
            self.right[1].weight.data = torch.from_numpy(bn3w)
            bn3b = np.load(os.path.join(dir, "bn{}_bias.npy".format(bn_num)))
            self.right[1].bias.data = torch.from_numpy(bn3b)
            bn3m = np.load(os.path.join(dir, "bn{}_mean.npy".format(bn_num)))
            self.right[1].running_mean.data = torch.from_numpy(bn3m)
            bn3v = np.load(os.path.join(dir, "bn{}_var.npy".format(bn_num)))
            self.right[1].running_var.data = torch.from_numpy(bn3v)
            bn_num += 1

        return conv_num, bn_num
    

class ResNet(nn.Module):
    '''
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
        
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer( 64, 64, 3)
        self.layer2 = self._make_layer( 64, 128, 4, stride=2)
        self.layer3 = self._make_layer( 128, 256, 6, stride=2)
        self.layer4 = self._make_layer( 256, 512, 3, stride=2)

        #分类用的全连接
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self,  inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
                nn.BatchNorm2d(outchannel))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        
        for _ in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return F.sigmoid(x)

    def save(self, path):
        conv_num = 0
        bn_num = 0

        if os.path.exists(path) == False:
            os.mkdir(path)

        conv1 = self.pre[0].weight.data.detach().cpu().numpy()
        np.save(os.path.join(path, "conv{}_weight.npy".format(str(conv_num))), conv1)
        conv_num += 1

        bn1w = self.pre[1].weight.data.detach().cpu().numpy()
        np.save(os.path.join(path, "bn{}_weight.npy".format(str(bn_num))), bn1w)
        bn1b = self.pre[1].bias.data.detach().cpu().numpy()
        np.save(os.path.join(path, "bn{}_bias.npy".format(str(bn_num))), bn1b)
        bn1m = self.pre[1].running_mean.data.detach().cpu().numpy()
        np.save(os.path.join(path, "bn{}_mean.npy".format(str(bn_num))), bn1m)
        bn1v = self.pre[1].running_var.data.detach().cpu().numpy()
        np.save(os.path.join(path, "bn{}_var.npy".format(str(bn_num))), bn1v)
        bn_num += 1

        for l in self.layer1:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer2:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer3:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer4:
            conv_num, bn_num = l.save(path, conv_num, bn_num)

        fcw = self.fc.weight.data.detach().cpu().numpy()
        np.save(os.path.join(path, "fc_weight.npy"), fcw)
        fcb = self.fc.bias.data.detach().cpu().numpy()
        np.save(os.path.join(path, "fc_bias.npy"), fcb)

    def load(self, path):
        conv_num = 0
        bn_num = 0

        conv1 = np.load(os.path.join(path, "conv{}_weight.npy".format(conv_num)))
        self.pre[0].weight.data = torch.from_numpy(conv1)
        conv_num += 1

        bn1w = np.load(os.path.join(path, "bn{}_weight.npy".format(bn_num)))
        self.pre[1].weight.data = torch.from_numpy(bn1w)
        bn1b = np.load(os.path.join(path, "bn{}_bias.npy".format(bn_num)))
        self.pre[1].bias.data = torch.from_numpy(bn1b)
        bn1m = np.load(os.path.join(path, "bn{}_mean.npy".format(bn_num)))
        self.pre[1].running_mean.data = torch.from_numpy(bn1m)
        bn1v = np.load(os.path.join(path, "bn{}_var.npy".format(bn_num)))
        self.pre[1].running_var.data = torch.from_numpy(bn1v)
        bn_num += 1

        for l in self.layer1:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer2:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer3:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer4:
            conv_num, bn_num = l.load(path, conv_num, bn_num)

        fcw = np.load(os.path.join(path, "fc_weight.npy"))
        self.fc.weight.data = torch.from_numpy(fcw)
        fcb = np.load(os.path.join(path, "fc_bias.npy"))
        self.fc.bias.data = torch.from_numpy(fcb)