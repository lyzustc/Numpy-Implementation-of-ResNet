import cv2
import numpy as np

def get_test_images(filename, image_w, image_h):
    with open(filename) as file:
        filelist = file.readlines()
    img_tensor = np.zeros([len(filelist), 3, image_w, image_h])
    label_tensor = np.zeros([len(filelist)])
    for i in range(len(filelist)):
        path, label = filelist[i].split(" ")
        img = cv2.imread(path)
        img = img[:,:,::-1].astype(np.float32)/255.0
        img_tensor[i] = img.transpose(2,0,1)
        label_tensor[i] = int(label)
    return img_tensor, label_tensor

def test(net, filename, image_w, image_h):
    images, labels = get_test_images(filename, image_w, image_h)
    infers = net.inference(images)
    return np.sum(infers == (labels-1)) / infers.shape[0]