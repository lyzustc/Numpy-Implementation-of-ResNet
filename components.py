import numpy as np

class layer:
    def __init__(self, in_channels, out_channels, kernel_h, kernel_w):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.kernel = np.zeros([out_channels, in_channels, kernel_h, kernel_w])
        self.bias = np.zeros([out_channels, 1])
    def forward(self):
        None
    def backward(self):
        None

class fl_sigmoid(layer):
    def __init__(self, in_channels, out_channels):
        layer.__init__(self, in_channels, out_channels, 1, 1)
        self.init_param()
    def init_param(self):
        self.kernel = self.kernel.reshape(self.out_channels, self.in_channels)
    def forward(self, in_tensor):
        self.in_tensor = in_tensor.reshape(in_tensor.shape[0], -1)
        self.out_tensor = np.dot(self.in_tensor, self.kernel.T) + self.bias.T
        self.out_tensor = 1.0 / (1.0 + np.exp(-self.out_tensor))
        return self.out_tensor
    def backward(self, out_diff_tensor, lr):
        nonlinear_diff = self.out_tensor * (1 - self.out_tensor) * out_diff_tensor
        kernel_diff = np.dot(nonlinear_diff.T, self.in_tensor).squeeze()
        bias_diff = np.sum(nonlinear_diff)
        self.in_diff_tensor = np.dot(nonlinear_diff, self.kernel).squeeze()
        self.kernel -= lr * kernel_diff
        self.bias -= lr * bias_diff