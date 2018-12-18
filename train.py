from data import *
from model import *
from test import *

class trainer:
    def __init__(self, model, dataset, lr):
        self.dataset = dataset
        self.net = model
        self.lr = lr
        self.cls_num = self.net.out_channels

    def iterate(self):
        images, labels = self.dataset.get_next_batch()

        out_tensor = self.net.forward(images)

        if self.cls_num > 1:
            one_hot_labels = np.eye(self.cls_num)[(labels-1).reshape(-1)].reshape(out_tensor.shape)
        else:
            one_hot_labels = (labels-1).reshape(out_tensor.shape)
            
        loss = np.sum(-one_hot_labels * np.log(out_tensor)-(1-one_hot_labels) * np.log(1 - out_tensor)) / self.dataset.batch_size
        out_diff_tensor = (out_tensor - one_hot_labels) / out_tensor / (1 - out_tensor) / self.dataset.batch_size
        
        self.net.backward(out_diff_tensor, self.lr)
        
        return loss

if __name__ == '__main__':
    a = single_layer_network(3,30)