from components import *

class single_layer_network:
    def __init__(self, in_channels, out_channels):
        self.fc_sigmoid = fc_sigmoid(in_channels, out_channels)
        self.out_channels = out_channels
    def forward(self, in_tensor):
        return self.fc_sigmoid.forward(in_tensor)
    def backward(self, out_diff_tensor, lr):
        self.fc_sigmoid.backward(out_diff_tensor, lr)
    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], self.out_channels)
        return np.argmax(out_tensor, axis=1)

class one_conv_network:
    def __init__(self, out_channels, image_h, image_w):
        self.conv = conv_layer(3, 8, 3, 3)
        self.pool = max_pooling(2)
        self.fc = fc_sigmoid(int(image_h/2)*int(image_w/2)*8, out_channels)
        self.out_channels = out_channels
    def forward(self, in_tensor):
        feature = self.pool.forward(self.conv.forward(in_tensor))
        pred = self.fc.forward(feature)
        return pred
    def backward(self, out_diff_tensor, lr):
        self.fc.backward(out_diff_tensor, lr)
        self.pool.backward(self.fc.in_diff_tensor)
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
        self.fc = fc_sigmoid(32*int(image_h/8)*int(image_w/8), out_channels)
        self.out_channels = out_channels

    def forward(self, in_tensor):
        feature1 = self.pool1.forward(self.conv1.forward(in_tensor))
        feature2 = self.pool2.forward(self.conv2.forward(feature1))
        feature3 = self.pool3.forward(self.conv3.forward(feature2))
        pred = self.fc.forward(feature3)
        
        return pred

    def backward(self, out_diff_tensor, lr):
        self.fc.backward(out_diff_tensor, lr)
        self.pool3.backward(self.fc.in_diff_tensor)
        self.conv3.backward(self.pool3.in_diff_tensor, lr)
        self.pool2.backward(self.conv3.in_diff_tensor)
        self.conv2.backward(self.pool2.in_diff_tensor, lr)
        self.pool1.backward(self.conv2.in_diff_tensor)
        self.conv1.backward(self.pool1.in_diff_tensor, lr)

    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], self.out_channels)
        return np.argmax(out_tensor, axis=1)

class vgg:

    def __init__(self, out_channels, img_h, img_w):
        self.conv1_1 = conv_layer(3, 64, 3, 3)
        self.conv1_2 = conv_layer(64, 64, 3, 3)
        self.pool1 = max_pooling(2)
        self.conv2_1 = conv_layer(64, 128, 3, 3)
        self.conv2_2 = conv_layer(128, 128, 3, 3)
        self.pool2 = max_pooling(2)
        self.conv3_1 = conv_layer(128, 256, 3, 3)
        self.conv3_2 = conv_layer(256, 256, 3, 3)
        self.conv3_3 = conv_layer(256, 256, 3, 3)
        self.pool3 = max_pooling(2)
        self.conv4_1 = conv_layer(256, 512, 3, 3)
        self.conv4_2 = conv_layer(512, 512, 3, 3)
        self.conv4_3 = conv_layer(512, 512, 3, 3)
        self.pool4 = max_pooling(2)
        self.fc1 = fc_layer(int(img_h/16)*int(img_w/16)*512, 4096)
        self.fc2 = fc_layer(4096, 4096)
        self.fc3 = fc_sigmoid(4096, out_channels)
        self.out_channels = out_channels
    
    def forward(self, in_tensor):
        feature1 = self.pool1.forward(self.conv1_2.forward(self.conv1_1.forward(in_tensor)))
        feature2 = self.pool2.forward(self.conv2_2.forward(self.conv2_1.forward(feature1)))
        feature3 = self.pool3.forward(self.conv3_3.forward(self.conv3_2.forward(self.conv3_1.forward(feature2))))
        feature4 = self.pool4.forward(self.conv4_3.forward(self.conv4_2.forward(self.conv4_1.forward(feature3))))
        pred = self.fc3.forward(self.fc2.forward(self.fc1.forward(feature4)))
        return pred

    def backward(self, out_diff_tensor, lr):
        self.fc3.backward(out_diff_tensor, lr)
        self.fc2.backward(self.fc3.in_diff_tensor, lr)
        self.fc1.backward(self.fc2.in_diff_tensor, lr)

        self.pool4.backward(self.fc1.in_diff_tensor)
        self.conv4_3.backward(self.pool4.in_diff_tensor, lr)
        self.conv4_2.backward(self.conv4_3.in_diff_tensor, lr)
        self.conv4_1.backward(self.conv4_2.in_diff_tensor, lr)

        self.pool3.backward(self.conv4_1.in_diff_tensor)
        self.conv3_3.backward(self.pool3.in_diff_tensor, lr)
        self.conv3_2.backward(self.conv3_3.in_diff_tensor, lr)
        self.conv3_1.backward(self.conv3_2.in_diff_tensor, lr)

        self.pool2.backward(self.conv3_1.in_diff_tensor)
        self.conv2_2.backward(self.pool2.in_diff_tensor, lr)
        self.conv2_1.backward(self.conv2_2.in_diff_tensor, lr)

        self.pool1.backward(self.conv2_1.in_diff_tensor)
        self.conv1_2.backward(self.pool1.in_diff_tensor, lr)
        self.conv1_1.backward(self.conv1_2.in_diff_tensor, lr)

    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], self.out_channels)
        return np.argmax(out_tensor, axis=1)