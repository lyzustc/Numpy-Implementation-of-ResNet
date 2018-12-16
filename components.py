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

class conv_layer(layer):
    
    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, same = True):
        layer.__init__(self, in_channels, out_channels, kernel_h, kernel_w)
        self.init_param()
        self.same = same
        
    def init_param(self):
        self.kernel += 1
        self.bias += 1
    
    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = np.zeros([batch_num, in_channels, in_h + 2*pad_h, in_w + 2*pad_w])
        padded[:, :, pad_h:pad_h+in_h, pad_w:pad_w+in_w] = in_tensor
        return padded
    
    @staticmethod
    def convolution(in_tensor, kernel):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_channels = kernel.shape[0]
        assert kernel.shape[1] == in_channels
        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]
        
        out_h = in_h - kernel_h + 1
        out_w = in_w - kernel_w + 1
        
        kernel = kernel.reshape(out_channels, -1)
        
        extend_in = np.zeros([in_channels*kernel_h*kernel_w, batch_num*out_h*out_w])
        for i in range(out_h):
            for j in range(out_w):
                part_in = in_tensor[:, :, i:i+kernel_h, j:j+kernel_w].reshape(batch_num, -1)
                extend_in[:, (i*out_w+j)*batch_num:(i*out_w+j+1)*batch_num] = part_in.T
        
        out_tensor = np.dot(kernel, extend_in)
        out_tensor = out_tensor.reshape(out_channels, out_h*out_w, batch_num)
        out_tensor = out_tensor.transpose(2,0,1).reshape(batch_num, out_channels, out_h, out_w) 
        
        return out_tensor
    
    def forward(self, in_tensor):
        if self.same:
            in_tensor = conv_layer.pad(in_tensor, int((self.kernel_h-1)/2), int((self.kernel_w-1)/2))
        
        self.in_tensor = in_tensor
        
        self.out_tensor = conv_layer.convolution(in_tensor, self.kernel)
        self.out_tensor += self.bias.reshape(1,self.out_channels,1,1)
        
        return self.out_tensor
    
    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        
        bias_diff = np.sum(out_diff_tensor, axis = (0,2,3)).reshape(self.bias.shape)
        
        kernel_diff = conv_layer.convolution(self.in_tensor.transpose(1,0,2,3), out_diff_tensor.transpose(1,0,2,3))
        kernel_diff = kernel_diff.transpose(1,0,2,3)
        
        padded = conv_layer.pad(out_diff_tensor, self.kernel_h-1, self.kernel_w-1)
        kernel_trans = self.kernel.reshape(self.out_channels, self.in_channels, self.kernel_h*self.kernel_w)
        kernel_trans = kernel_trans[:,:,::-1].reshape(self.kernel.shape)
        self.in_diff_tensor = conv_layer.convolution(padded, kernel_trans.transpose(1,0,2,3))
        if self.same:
            pad_h = int((self.kernel_h-1)/2)
            pad_w = int((self.kernel_w-1)/2)
            self.in_diff_tensor = self.in_diff_tensor[:, :, pad_h:-pad_h, pad_w:-pad_w]
            
        self.bias -= lr * bias_diff
        self.kernel -= lr * kernel_diff