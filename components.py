import numpy as np
import os

class fc_sigmoid:

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_param()

    def init_param(self):
        self.kernel = np.random.uniform(
            low = -np.sqrt(6.0/(self.out_channels + self.in_channels)),
            high = np.sqrt(6.0/(self.in_channels + self.out_channels)),
            size = (self.out_channels, self.in_channels)
        )
        self.bias = np.zeros([self.out_channels])

    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        self.in_tensor = in_tensor.reshape(in_tensor.shape[0], -1)
        assert self.in_tensor.shape[1] == self.kernel.shape[1]
        self.out_tensor = np.dot(self.in_tensor, self.kernel.T) + self.bias.T
        self.out_tensor = 1.0 / (1.0 + np.exp(-self.out_tensor))
        return self.out_tensor

    def backward(self, out_diff_tensor, lr):
        nonlinear_diff = self.out_tensor * (1 - self.out_tensor) * out_diff_tensor
        kernel_diff = np.dot(nonlinear_diff.T, self.in_tensor).squeeze()
        bias_diff = np.sum(nonlinear_diff, axis=0).reshape(self.bias.shape)
        self.in_diff_tensor = np.dot(nonlinear_diff, self.kernel).reshape(self.shape)
        self.kernel -= lr * kernel_diff
        self.bias -= lr * bias_diff

    def save(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)

        np.save(os.path.join(path, "fc_weight.npy"), self.kernel)
        np.save(os.path.join(path, "fc_bias.npy"), self.bias)

    def load(self, path):
        assert os.path.exists(path)

        self.kernel = np.load(os.path.join(path, "fc_weight.npy"))
        self.bias = np.load(os.path.join(path, "fc_bias.npy"))



class conv_layer:

    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, same = True, relu = True, shift = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.same = same
        self.relu = relu
        self.shift = shift

        self.init_param()

    def init_param(self):
        self.kernel = np.random.uniform(
            low = -np.sqrt(6.0/(self.out_channels + self.in_channels * self.kernel_h * self.kernel_w)),
            high = np.sqrt(6.0/(self.in_channels + self.out_channels * self.kernel_h * self.kernel_w)),
            size = (self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        )
        self.bias = np.zeros([self.out_channels, 1]) if self.shift else None

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

        if self.shift:
            self.out_tensor += self.bias.reshape(1,self.out_channels,1,1)

        if self.relu:
            self.out_tensor[self.out_tensor < 0] = 0

        return self.out_tensor
    
    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.out_tensor.shape
        
        if self.relu:
            out_diff_tensor[self.out_tensor <= 0] = 0

        if self.shift:
            bias_diff = np.sum(out_diff_tensor, axis = (0,2,3)).reshape(self.bias.shape)
            self.bias -= lr * bias_diff

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
            
        self.kernel -= lr * kernel_diff

    def save(self, path, conv_num):
        if os.path.exists(path) == False:
            os.mkdir(path)

        np.save(os.path.join(path, "conv{}_weight.npy".format(conv_num)), self.kernel)
        if self.shift:
            np.save(os.path.join(path, "conv{}_bias.npy".format(conv_num)), self.bias)
        
        return conv_num + 1

    def load(self, path, conv_num):
        assert os.path.exists(path)

        self.kernel = np.load(os.path.join(path, "conv{}_weight.npy".format(conv_num)))
        if self.shift:
            self.bias = np.load(os.path.join(path, "conv{}_bias.npy").format(conv_num))
        
        return conv_num + 1



class max_pooling:
    
    def __init__(self, stride):
        self.stride = stride

    def forward(self, in_tensor):
        assert in_tensor.shape[2] % self.stride == 0
        assert in_tensor.shape[3] % self.stride == 0
        self.in_tensor = in_tensor

        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_h = int(in_h/self.stride)
        out_w = int(in_w/self.stride)

        extend_in = in_tensor.reshape(batch_num, in_channels, out_h, self.stride, out_w, self.stride)
        extend_in = extend_in.transpose(0,1,2,4,3,5).reshape(batch_num, in_channels, out_h, out_w, -1)

        self.maxindex = np.argmax(extend_in, axis = 4)
        out_tensor = extend_in.max(axis = 4)

        return out_tensor

    def backward(self, out_diff_tensor):
        batch_num = out_diff_tensor.shape[0]
        in_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        
        
        out_diff_tensor = out_diff_tensor.reshape(-1)
        self.maxindex = self.maxindex.reshape(-1)
        
        in_diff_tensor = np.zeros([batch_num*in_channels*out_h*out_w, self.stride*self.stride])
        in_diff_tensor[range(batch_num*in_channels*out_h*out_w), self.maxindex] = out_diff_tensor
        in_diff_tensor = in_diff_tensor.reshape(batch_num, in_channels, out_h, out_w, self.stride, self.stride)
        in_diff_tensor = in_diff_tensor.transpose(0,1,2,4,3,5)
        in_diff_tensor = in_diff_tensor.reshape(batch_num, in_channels, out_h*self.stride, out_w*self.stride)
        
        self.in_diff_tensor = in_diff_tensor



class average_pooling:
    
    def __init__(self, stride):
        self.stride = stride

    def forward(self, in_tensor):
        assert in_tensor.shape[2] % self.stride == 0
        assert in_tensor.shape[3] % self.stride == 0
        self.in_tensor = in_tensor

        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_h = int(in_h/self.stride)
        out_w = int(in_w/self.stride)

        extend_in = in_tensor.reshape(batch_num, in_channels, out_h, self.stride, out_w, self.stride)
        extend_in = extend_in.transpose(0,1,2,4,3,5).reshape(batch_num, in_channels, out_h, out_w, -1)

        out_tensor = extend_in.mean(axis = 4)

        return out_tensor

    def backward(self, out_diff_tensor):
        batch_num = out_diff_tensor.shape[0]
        in_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        
        
        out_diff_tensor = out_diff_tensor.reshape(-1, 1)
        
        in_diff_tensor = np.zeros([batch_num*in_channels*out_h*out_w, self.stride*self.stride])
        in_diff_tensor += out_diff_tensor / self.stride / self.stride
        in_diff_tensor = in_diff_tensor.reshape(batch_num, in_channels, out_h, out_w, self.stride, self.stride)
        in_diff_tensor = in_diff_tensor.transpose(0,1,2,4,3,5)
        in_diff_tensor = in_diff_tensor.reshape(batch_num, in_channels, out_h*self.stride, out_w*self.stride)
        
        self.in_diff_tensor = in_diff_tensor



class bn_layer:

    def __init__(self, neural_num, moving_rate, is_train = True):
        self.gamma = np.random.uniform(low=0, high=1, size=neural_num)
        self.bias = np.zeros([neural_num])
        self.moving_avg = np.zeros([neural_num])
        self.moving_var = np.ones([neural_num])
        self.neural_num = neural_num
        self.moving_rate = moving_rate
        self.is_train = is_train
        self.epsilon = 1e-5

    def forward(self, in_tensor):
        assert in_tensor.shape[1] == self.neural_num

        self.in_tensor = in_tensor

        if self.is_train:
            mean = in_tensor.mean(axis=(0,2,3))
            var = in_tensor.var(axis=(0,2,3))
            self.moving_avg = mean * self.moving_rate + (1 - self.moving_rate) * self.moving_avg
            self.moving_var = var * self.moving_rate + (1 - self.moving_rate) * self.moving_var
            self.var = var
        else:
            mean = self.moving_avg
            var = self.moving_var

        normalized = (in_tensor - mean.reshape(1,-1,1,1)) / np.sqrt(var.reshape(1,-1,1,1)+self.epsilon)
        out_tensor = self.gamma.reshape(1,-1,1,1) * normalized + self.bias.reshape(1,-1,1,1)

        return out_tensor

    def backward(self, out_diff_tensor, lr):
        assert out_diff_tensor.shape == self.in_tensor.shape
        assert self.is_train
        
        self.in_diff_tensor = self.gamma.reshape(1,-1,1,1) / np.sqrt(self.var.reshape(1,-1,1,1)+self.epsilon) * out_diff_tensor
        
        gamma_diff = self.in_tensor / np.sqrt(self.var.reshape(1,-1,1,1)+self.epsilon) * out_diff_tensor
        self.gamma -= lr * gamma_diff

        bias_diff = np.sum(out_diff_tensor, axis = (0,2,3))
        self.bias -= lr * bias_diff

    def save(self, path, bn_num):
        if os.path.exists(path) == False:
            os.mkdir(path)

        np.save(os.path.join(path, "bn{}_weight.npy".format(bn_num)), self.gamma)
        np.save(os.path.join(path, "bn{}_bias.npy".format(bn_num)), self.bias)
        np.save(os.path.join(path, "bn{}_mean.npy".format(bn_num)), self.moving_avg)
        np.save(os.path.join(path, "bn{}_var.npy".format(bn_num)), self.moving_var)

        return bn_num + 1

    def load(self, path, bn_num):
        assert os.path.exists(path)

        self.gamma = np.load(os.path.join(path, "bn{}_weight.npy".format(bn_num)))
        self.bias = np.load(os.path.join(path, "bn{}_bias.npy".format(bn_num)))
        self.moving_avg = np.load(os.path.join(path, "bn{}_mean.npy".format(bn_num)))
        self.moving_var = np.load(os.path.join(path, "bn{}_var.npy".format(bn_num)))

        return bn_num + 1