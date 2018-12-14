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
        out_tensor = self.forward(in_tensor)
        return np.argmax(out_tensor) + 1