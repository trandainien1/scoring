import timm_detach
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gc
from Datasets.PascalVOC import PascalVOC2012Dataset_remove_multi_class
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from AGCAM import AGCAM
import cv2

MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
NAME = "Pascal base ours"

model_path = os.path.join("model_parameters", "pascal_0117_epoch52.pt")
save_root = os.path.join('/home/NAS_mount/sbim/ViT_figures/PascalVOC2012/Ours')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == DEVICE:
    gc.collect()
    torch.cuda.empty_cache()
print("device: " +  device)

batch_size = 1
img_size = 224

classes = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair","cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
root_dir = '/home/NAS_mount/sbim/Dataset/Pascal VOC 2012/trainval/VOCdevkit/VOC2012'
mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

unnormalize = transforms.Compose([
    transforms.Normalize([0., 0., 0.], [1/0.26753769276329037, 1/0.2638145880487105, 1/0.2776826934044154]),
    transforms.Normalize([-0.457342265910642, -0.4387686270106377, -0.4073427106250871], [1., 1., 1.,])
])

valid_transform = transforms.Compose([transforms.Resize((img_size, img_size)),          
                                      transforms.ToTensor(), 
                                    transforms.Normalize(mean = mean, std = std),
                                      ])


dataset_val = PascalVOC2012Dataset_remove_multi_class(root=root_dir, train=False, transforms=valid_transform)
valid_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)


model = timm_detach.create_model(MODEL, pretrained=True, num_classes=20).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
method = AGCAM(model)




with torch.enable_grad():
    for data in tqdm(valid_loader):
        image = data['image'].cuda()
        label = data['label'].cuda()
        prediction, heatmap = method.generate(image)
        if prediction!=label:
            continue
        # heatmap = heatmap.reshape(1, 1, 14, 14)
        resize = transforms.Resize((224, 224))
        heatmap = resize(heatmap[0])
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.transpose(heatmap, (1, 2, 0))
        image = unnormalize(image)
        # print(image.max(), image.min())
        image = image.detach().cpu().numpy()[0]
        image = np.transpose(image, (1, 2, 0))
        image_name = data['filename'][0] +"_"+ classes[label] + ".jpeg"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.uint8(image*255)
        heatmap = np.uint8(heatmap*255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        cv2.imwrite(os.path.join(save_root, image_name), image)




        