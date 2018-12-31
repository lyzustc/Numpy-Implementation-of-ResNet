import numpy as np

class layer:

    def __init__(self, in_channels, out_channels, kernel_h, kernel_w):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.kernel = np.zeros([out_channels, in_channels, kernel_h, kernel_w])
        self.bias = np.zeros([out_channels, 1])

class fc_layer(layer):

    def __init__(self, in_channels, out_channels, relu=True, shift=True):
        layer.__init__(self, in_channels, out_channels, 1, 1)
        self.relu = relu
        self.shift = shift
        self.init_param()

    def init_param(self):
        self.kernel = np.random.uniform(
            low = -np.sqrt(6.0/(self.out_channels + self.in_channels)),
            high = np.sqrt(6.0/(self.in_channels + self.out_channels)),
            size = (self.out_channels, self.in_channels)
        )

    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        self.in_tensor = in_tensor.reshape(in_tensor.shape[0], -1)
        assert self.in_tensor.shape[1] == self.kernel.shape[1]
        self.out_tensor = np.dot(self.in_tensor, self.kernel.T) 
        
        if self.shift:
            self.out_tensor += self.bias.T

        if self.relu:
            self.out_tensor[self.out_tensor < 0] = 0
        
        return self.out_tensor

    def backward(self, out_diff_tensor, lr):

        assert out_diff_tensor.shape == self.out_tensor.shape
        
        if self.relu:
            out_diff_tensor[self.out_tensor <= 0] = 0

        kernel_diff = np.dot(out_diff_tensor.T, self.in_tensor).squeeze()
        self.in_diff_tensor = np.dot(out_diff_tensor, self.kernel).reshape(self.shape)
        self.kernel -= lr * kernel_diff

        if self.shift:
            bias_diff = np.sum(out_diff_tensor, axis=0).reshape(self.bias.shape)
            self.bias -= lr * bias_diff

class fc_sigmoid(layer):

    def __init__(self, in_channels, out_channels):
        layer.__init__(self, in_channels, out_channels, 1, 1)
        self.init_param()

    def init_param(self):
        self.kernel = np.random.uniform(
            low = -np.sqrt(6.0/(self.out_channels + self.in_channels)),
            high = np.sqrt(6.0/(self.in_channels + self.out_channels)),
            size = (self.out_channels, self.in_channels)
        )

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

class conv_layer(layer):

    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, same = True, relu = True, shift = True):
        layer.__init__(self, in_channels, out_channels, kernel_h, kernel_w)
        self.init_param()
        self.same = same
        self.relu = relu
        self.shift = shift

    def init_param(self):
        self.kernel = np.random.uniform(
            low = -np.sqrt(6.0/(self.out_channels + self.in_channels * self.kernel_h * self.kernel_w)),
            high = np.sqrt(6.0/(self.in_channels + self.out_channels * self.kernel_h * self.kernel_w)),
            size = self.kernel.shape
        )
        #self.kernel += 1

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