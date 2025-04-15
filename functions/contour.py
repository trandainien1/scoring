import torch 
from torch import Tensor
import cv2
import torch.nn as nn
import numpy as np


def get_largest_segment(mask, threshold=None):
    
    upsample = nn.Upsample(224, mode = 'bilinear', align_corners=False)
    mask = upsample(mask)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    
    mask = mask.detach().cpu().numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    if threshold==None:
        threshold = mask.mean()
    mask[mask>threshold]=1
    mask[mask<=threshold]=0
    mask = (mask*255).astype(np.uint8)
    # ret, thr = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    max_area = 0
    max_idx = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if max_area<area:
            max_area = area
            max_idx = i

    x, y, w, h = cv2.boundingRect(contours[max_idx])
    xmin = x
    ymin = y
    xmax = x+w
    ymax = y+h
    bnd_box = torch.tensor([xmin, ymin, xmax, ymax]).unsqueeze(0)
    return bnd_box
    




