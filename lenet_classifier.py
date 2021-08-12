import torch
import torch.nn as nn
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np

from model import LeNet

def transform(img):
    # Input: img <class 'numpy.ndarray'> (h, w, 3)
    # Output: out <class 'torch.Tensor'> torch.Size([1, 1, 32, 32]) torch.float32
    
    img_gray = img[:, :, 0]
    out = np.zeros((1, 1, 32, 32))
    hs = img.shape[0] // 32
    ws = img.shape[1] // 32
    
    for i in range(32):
        for j in range(32):
            out[0, 0, i, j] = img_gray[i * hs, j * ws]
            
    img_t = torch.tensor(out, dtype=torch.float32)
    
    # background (in white) = 0
    # foreground (in black) <= 1.0
    return torch.ones(1, 1, 32, 32) - img_t

class LeNetClassifier():
    def __init__(self, weight_path='./weights/2021-08-10-19-19.pth'):
        save_info = torch.load(weight_path)
        self.model = LeNet()
        self.model.load_state_dict(save_info["model"])
        self.model.eval()
        
    def run(self, img):
        # Input: img <class 'numpy.ndarray'> (h, w, 3)
        # Output: label: <class 'int'>
        #         confs: <class 'torch.Tensor'> torch.Size([1, 10]) torch.float32
        
        img_t = transform(img)
        with torch.no_grad():
            confs = self.model(img_t)
            label = int(confs.max(1).indices)
        
        return label, confs
        
if __name__ == '__main__':
    classifier = LeNetClassifier()
    
    img = imgplt.imread('./exp/0.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/1.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/2.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/3.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/4.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/5.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/6.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/7.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/8.png')
    label, confs = classifier.run(img)
    print('label: ', label)
    
    img = imgplt.imread('./exp/9.png')
    label, confs = classifier.run(img)
    print('label: ', label)
