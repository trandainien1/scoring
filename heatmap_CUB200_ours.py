import os
import timm_detach
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gc
from Datasets.CUB200 import CUBDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from AGCAM import AGCAM
import random
import cv2

MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
NAME = "CUB200 Attention ours"

model_path = os.path.join("model_parameters", "CUB_1002_epoch467.pt")
save_root = os.path.join('/home/NAS_mount/sbim/ViT_figures/CUB200/Ours')



device = 'cuda' if torch.cuda.is_available() else 'cpu'
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(777)
if device == DEVICE:
    gc.collect()
    torch.cuda.empty_cache()
print("device: " +  device)

batch_size = 1
img_size = 224


mean = [0.4859, 0.4996, 0.4318]
std = [0.1750, 0.1739, 0.1859]

unnormalize = transforms.Compose([
    transforms.Normalize(mean=[-0.4859/0.1750, -0.4996/0.1739, -0.4318/0.1859], std = [1/0.1750, 1/0.1739, 1/0.1859])
])

valid_transform = transforms.Compose([transforms.Resize((img_size, img_size)),          
                                      transforms.ToTensor(), 
                                    transforms.Normalize(mean = mean, std = std),
                                      ])
validset = CUBDataset(
    root = '/home/NAS_mount/sbim/Dataset/CUB_200/CUB_200_2011',
    train= False,
    transforms = valid_transform,
)
valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)      

model = timm_detach.create_model(MODEL, pretrained=True, num_classes=200).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
# if args.method =="single":
method = AGCAM(model)




with torch.enable_grad():
    for data in tqdm(valid_loader):
        image = data['image'].cuda()
        label = data['label'].cuda()
        bnd_box = data['bnd_box'].cuda().squeeze(0)

        prediction, heatmap = method.generate(image)
        if prediction!=label:
            continue
        
        resize = transforms.Resize((224, 224))
        heatmap = resize(heatmap[0])
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.transpose(heatmap, (1, 2, 0))
        image = unnormalize(image)
        image = image.detach().cpu().numpy()[0]
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.uint8(image*255)
        heatmap = np.uint8(heatmap*255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        image_name = data['filename'][0].split('/')[-1] +"_"+ str(label.item()) + ".jpeg"
        cv2.imwrite(os.path.join(save_root, image_name), image)
