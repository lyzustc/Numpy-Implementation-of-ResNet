from components import *

class single_layer_network:
    def __init__(self, in_channels, out_channels):
        self.fl_sigmoid = fl_sigmoid(in_channels, out_channels)
        self.out_channels = out_channels
    def forward(self, in_tensor):
        return self.fl_sigmoid.forward(in_tensor)
    def backward(self, out_diff_tensor, lr):
        self.fl_sigmoid.backward(out_diff_tensor, lr)
    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], self.out_channels)
        return np.argmax(out_tensor, axis=1)

class one_conv_network:
    def __init__(self, out_channels, image_h, image_w):
        self.conv = conv_layer(3, 8, 3, 3)
        self.pool = max_pooling(2)
        self.fl = fl_sigmoid(int(image_h/2)*int(image_w/2)*8, out_channels)
        self.out_channels = out_channels
    def forward(self, in_tensor):
        feature = self.pool.forward(self.conv.forward(in_tensor))
        prev = self.fl.forward(feature)
        return prev
    def backward(self, out_diff_tensor, lr):
        self.fl.backward(out_diff_tensor, lr)
        self.pool.backward(self.fl.in_diff_tensor)
        self.conv.backward(self.pool.in_diff_tensor, lr)
    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], self.out_channels)
        return np.argmax(out_tensor, axis=1)

class three_conv_network:
    def __init__(self, out_channels, image_h, image_w):
        self.conv1 = conv_layer(3, 8, 3, 3)
        self.pool1 = max_pooling(2)
        self.conv2 = conv_layer(8, 16, 3, 3)
        self.pool2 = max_pooling(2)
        self.conv3 = conv_layer(16, 32, 3, 3)
        self.pool3 = max_pooling(2)
        self.fl = fl_sigmoid(32*int(image_h/8)*int(image_w/8), out_channels)
        self.out_channels = out_channels

    def forward(self, in_tensor):
        feature1 = self.pool1.forward(self.conv1.forward(in_tensor))
        feature2 = self.pool2.forward(self.conv2.forward(feature1))
        feature3 = self.pool3.forward(self.conv3.forward(feature2))
        prev = self.fl.forward(feature3)
        
        return prev

    def backward(self, out_diff_tensor, lr):
        self.fl.backward(out_diff_tensor, lr)
        self.pool3.backward(self.fl.in_diff_tensor)
        self.conv3.backward(self.pool3.in_diff_tensor, lr)
        self.pool2.backward(self.conv3.in_diff_tensor)
        self.conv2.backward(self.pool2.in_diff_tensor, lr)
        self.pool1.backward(self.conv2.in_diff_tensor)
        self.conv1.backward(self.pool1.in_diff_tensor, lr)

    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], self.out_channels)
        return np.argmax(out_tensor, axis=1)