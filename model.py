from components import *

class ResBlock:

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        self.path1 = [
            conv_layer(in_channels, out_channels, 3, 3, stride = stride, shift=False),
            bn_layer(out_channels),
            relu(),
            conv_layer(out_channels, out_channels, 3, 3, shift=False),
            bn_layer(out_channels)
        ]
        self.path2 = shortcut
        self.relu = relu()
    
    def train(self):
        self.path1[1].train()
        self.path1[4].train()
        if self.path2 is not None:
            self.path2[1].train()

    def eval(self):
        self.path1[1].eval()
        self.path1[4].eval()
        if self.path2 is not None:
            self.path2[1].eval()

    def forward(self, in_tensor):
        x1 = in_tensor.copy()
        x2 = in_tensor.copy()

        for l in self.path1:
            x1 = l.forward(x1)
        if self.path2 is not None:
            for l in self.path2:
                x2 = l.forward(x2)
        self.out_tensor = self.relu.forward(x1+x2)

        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        assert self.out_tensor.shape == out_diff_tensor.shape
        x1 = out_diff_tensor.copy()
        x2 = out_diff_tensor.copy()

        for l in range(len(self.path1)):
            self.path1[-l].backward(x1, lr)
            x1 = self.path1[-l].in_diff_tensor

        if self.path2 is not None:
            for l in range(len(self.path2)):
                self.path1[-l].backward(x2, lr)
                x2 = self.path1[-l].in_diff_tensor

        self.in_diff_tensor = x1 + x2

    def save(self, path, conv_num, bn_num):
        conv_num = self.path1[0].save(path, conv_num)
        bn_num = self.path1[1].save(path, bn_num)
        conv_num = self.path1[3].save(path, conv_num)
        bn_num = self.path1[4].save(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].save(path, conv_num)
            bn_num = self.path2[1].save(path, bn_num)

        return conv_num, bn_num

    def load(self, path, conv_num, bn_num):
        conv_num = self.path1[0].load(path, conv_num)
        bn_num = self.path1[1].load(path, bn_num)
        conv_num = self.path1[3].load(path, conv_num)
        bn_num = self.path1[4].load(path, bn_num)

        if self.path2 is not None:
            conv_num = self.path2[0].load(path, conv_num)
            bn_num = self.path2[1].load(path, bn_num)

        return conv_num, bn_num



class resnet34:
    
    def __init__(self, num_classes):
        self.pre = [
            conv_layer(3, 64, 7, 7, stride=2, shift=False),
            bn_layer(64),
            relu(),
            max_pooling(3,3,2,same=True)
        ]
        self.layer1 = self.stack_ResBlock(64, 64, 3, 1)
        self.layer2 = self.stack_ResBlock(64, 128, 4, 2)
        self.layer3 = self.stack_ResBlock(128, 256, 6, 2)
        self.layer4 = self.stack_ResBlock(256, 512, 3, 2)
        self.avg = global_average_pooling()
        self.fc = fc_sigmoid(512, num_classes)

    def train(self):
        for l in self.layer1:
            l.train()
        for l in self.layer2:
            l.train()
        for l in self.layer3:
            l.train()
        for l in self.layer4:
            l.train()

    def eval(self):
        for l in self.layer1:
            l.eval()
        for l in self.layer2:
            l.eval()
        for l in self.layer3:
            l.eval()
        for l in self.layer4:
            l.eval()

    def stack_ResBlock(self, in_channels, out_channels, block_num, stride):
        shortcut = [
            conv_layer(in_channels, out_channels, 1, 1, stride=stride, shift=False),
            bn_layer(out_channels)
        ]
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride, shortcut=shortcut))

        for _ in range(block_num-1):
            layers.append(ResBlock(out_channels, out_channels))

        return layers

    def forward(self, in_tensor):
        x = in_tensor
        for l in self.pre:
            x = l.forward(x)
        for l in self.layer1:
            x = l.forward(x)
        for l in self.layer2:
            x = l.forward(x)
        for l in self.layer3:
            x = l.forward(x)
        for l in self.layer4:
            x = l.forward(x)
        x = self.avg.forward(x)
        out_tensor = self.fc.forward(x)
        
        return out_tensor

    def backward(self, out_diff_tensor, lr):
        x = out_diff_tensor
        self.fc.backward(x, lr)
        x = self.fc.in_diff_tensor
        self.avg.backward(x, lr)
        x = self.avg.in_diff_tensor
        for l in self.layer4:
            l.backward(x, lr)
            x = l.in_diff_tensor
        for l in self.layer3:
            l.backward(x, lr)
            x = l.in_diff_tensor
        for l in self.layer2:
            l.backward(x, lr)
            x = l.in_diff_tensor
        for l in self.layer1:
            l.backward(x, lr)
            x = l.in_diff_tensor
        for l in self.pre:
            l.backward(x, lr)
            x = l.in_diff_tensor
    
    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], -1)
        return np.argmax(out_tensor, axis=1)

    def save(self, path):
        conv_num = 0
        bn_num = 0

        conv_num = self.pre[0].save(path, conv_num)
        bn_num = self.pre[1].save(path, bn_num)

        for l in self.layer1:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer2:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer3:
            conv_num, bn_num = l.save(path, conv_num, bn_num)
        for l in self.layer4:
            conv_num, bn_num = l.save(path, conv_num, bn_num)

        self.fc.save(path)

    def load(self, path):
        conv_num = 0
        bn_num = 0

        conv_num = self.pre[0].load(path, conv_num)
        bn_num = self.pre[1].load(path, bn_num)

        for l in self.layer1:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer2:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer3:
            conv_num, bn_num = l.load(path, conv_num, bn_num)
        for l in self.layer4:
            conv_num, bn_num = l.load(path, conv_num, bn_num)

        self.fc.load(path)