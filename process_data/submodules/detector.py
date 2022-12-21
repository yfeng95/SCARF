import numpy as np
import torch
import argparse
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import os
from tqdm import tqdm
import shutil

'''
For cropping body:
1. using bbox from objection detectors
2. calculate bbox from body joints regressor
object detectors:
    know body object from label number
    https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    label for peopel: 1
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
'''
import os
import os.path as osp
import sys
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
# from PIL import Image

class FasterRCNN(object):
    ''' detect body
    '''
    def __init__(self, device='cuda:0'):  
        '''
        https://pytorch.org/docs/stable/torchvision/models.html#faster-r-cnn
        '''
        import torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def run(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.5)
        if len(inds) < 1 or inds.sum()<0.5:
            return None
        else:
            # if inds.sum()<0.5:
            #     inds = (prediction['labels']==1)*(prediction['scores']>0.05)
            bbox = prediction['boxes'][inds][0].cpu().numpy()
            return bbox
    
    @torch.no_grad()
    def run_multi(self, input):
        '''
        input: 
            The input to the model is expected to be a list of tensors, 
            each of shape [C, H, W], one for each image, and should be in 0-1 range. 
            Different images can have different sizes.
        return: 
            detected box, [x1, y1, x2, y2]
        '''
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels']==1)*(prediction['scores']>0.9)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds].cpu().numpy()
            return bbox