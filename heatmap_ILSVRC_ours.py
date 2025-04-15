import timm_detach
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gc
from Datasets.ILSVRC import ImageNetDataset
from AGCAM import AGCAM
from tqdm import tqdm
from functions.multi_object_mask import getBoudingBox_multi, box_to_seg
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from Datasets.ILSVRC_classes import classes
import cv2

MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
NAME = "ILSVRC base ours"

model_path = os.path.join("model_parameters", "jx_vit_base_p16_224-80ecf9dd.pth")
save_root = os.path.join('/home/NAS_mount/sbim/ViT_figures/ILSVRC2012/Ours')



device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == DEVICE:
    gc.collect()
    torch.cuda.empty_cache()
print("device: " +  device)

batch_size = 1
img_size = 224



test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

unnormalize = transforms.Compose([
    transforms.Normalize([0., 0., 0.], [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.,])
])

validateset = ImageNetDataset(
    img_dir = "/home/NAS_mount/sbim/Dataset/ILSVRC/Data/CLS-LOC/val",
    annotation_dir = "/home/NAS_mount/sbim/Dataset/ILSVRC/Annotations/CLS-LOC/val",
    transforms=test_transform,
)

validloader = DataLoader(
    dataset = validateset,
    batch_size=batch_size,
    shuffle = False,
)


model = timm_detach.create_model(MODEL, pretrained=True, num_classes=1000).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
# if args.method =="single":
method = AGCAM(model)



with torch.enable_grad():
    for data in tqdm(validloader):
        image = data['image'].cuda()
        label = data['label'].cuda()
        bnd_box = data['bnd_box'].cuda().squeeze(0)
        # print(data['filename'])
        # print(image.shape) #[1, 3, 224, 224]
        # print(label.shape) #[1]
        # print(bnd_box.shape) #[n, 4]

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
        image_name = data['filename'][0] +"_"+ classes[label.item()] + ".jpeg"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.uint8(image*255)
        heatmap = np.uint8(heatmap*255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        cv2.imwrite(os.path.join(save_root, image_name), image)
