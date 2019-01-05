import cv2
import numpy as np
import os
from model import *

def get_demo_images(filename, image_path, image_h=64, image_w=64):
    with open(filename) as file:
        filelist = file.readlines()
    img_tensor = np.zeros([len(filelist), 3, image_h, image_w])
    label_tensor = np.zeros([len(filelist)],dtype=np.int32)
    for i in range(len(filelist)):
        num, label = filelist[i].split(" ")
        #img = cv2.imread(os.path.join(image_path, "{}.JPEG".format(str(num))))
        img = cv2.imread(num)
        img = img[:,:,::-1].astype(np.float32)
        img_tensor[i] = img.transpose(2,0,1)
        label_tensor[i] = int(label)
    return img_tensor, label_tensor

def demo(model_path, filename, image_path, image_h=64, image_w=64):
    net = resnet34(20)
    net.load(model_path)
    net.eval()
    images, labels = get_demo_images(filename, image_path, image_h, image_w)
    top3_pred = np.zeros([images.shape[0],3],dtype=np.int32)
    
    num_accurate = 0
    for i in range(images.shape[0]):
        out = net.forward(images[i].reshape(1,3,image_h,image_w)).reshape(-1)
        order = np.argsort(out)[::-1]
        if order[0] < 0.5:
            top3_pred[i,0] = 21
            top3_pred[i,1:3] = order[0:2]+1
        else:
            top3_pred[i,0:3] = order[0:3]+1
        print("Top 3 prediction: {}  {}  {} || truth: {}".format(top3_pred[i,0], top3_pred[i,1], top3_pred[i,2], labels[i]))
        if labels[i] == top3_pred[i,0] or labels[i] == top3_pred[i,1] or labels[i] == top3_pred[i,2]:
            num_accurate += 1

    print("Among {} images, {} images are classified correctly.".format(images.shape[0], num_accurate))
    print("The accurate rate is {}".format(num_accurate/images.shape[0]))

if __name__ == "__main__":
    model_path = "model"
    filename = "test.txt"
    image_path = "test"
    demo(model_path, filename, image_path)
