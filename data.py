import cv2
import random
import numpy as np
import os

class dataloader:
    def __init__(self, filename, batch_size, image_w, image_h):
        with open(filename) as file:
            self.datalist = file.readlines()
        #random.shuffle(self.datalist)
        self.batch_size = batch_size
        self.len = len(self.datalist)
        self.index = 0
        self.image_w = image_w
        self.image_h = image_h
    def reset(self):
        self.index = 0
        #random.shuffle(self.datalist)
    def get_trans_img(self, path):
        img = cv2.imread(path)
        img = (img[:,:,::-1].astype(np.float32))/255
        img = img.transpose([2,0,1])
        return img
    def get_next_batch(self):
        if self.index + self.batch_size >= self.len:
            self.reset()
        images = np.zeros([self.batch_size, 3, self.image_w, self.image_h],dtype=np.float32)
        labels = np.zeros([self.batch_size],dtype=np.int32)
        for i in range(self.batch_size):
            path, label = self.datalist[i + self.index].split(' ')
            images[i] = self.get_trans_img(path)
            labels[i] = int(label)
        self.index += self.batch_size
        return images,labels