from data import *
from model import *
from test import *
import matplotlib.pyplot as plt

class trainer:
    def __init__(self, model, dataset, init_lr):
        self.dataset = dataset
        self.net = model
        self.lr = init_lr
        self.cls_num = self.net.out_channels

    def set_lr(self, lr):
        self.lr = lr

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
    batch_size = 32
    image_h = 64
    image_w = 64
    dataset = dataloader("train.txt", batch_size, image_h, image_w)

    model = resnet34(20)

    init_lr = 0.01
    train = trainer(model, dataset, init_lr)
    loss = []
    accurate = []
    temp = 0

    model.train()
    for i in range(25000):
        temp += train.iterate().numpy()
        if i % 10 == 0 and i != 0:
            loss.append(temp / 10)
            print("iteration = {} || loss = {}".format(str(i), str(temp/10)))
            temp = 0
            if i % 100 == 0:
                model.eval()
                accurate.append(test(model, "test.txt", image_h, image_w))
                model.save("model/model{}".format(i))
                model.train()
        plt.figure(figsize=(10,5))       
        plt.subplot(1,2,1)
        plt.plot(loss)
        plt.subplot(1,2,2)
        plt.plot(accurate)
        plt.show()

        if i == 15000:
            train.set_lr(0.001)