import timm_detach
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gc
from Datasets.ILSVRC import ImageNetDataset
import argparse
from AGCAM import AGCAM
from tqdm import tqdm
from functions.multi_object_mask import getBoudingBox_multi, box_to_seg
from torch.utils.data import DataLoader
import os
import numpy as np
import random

MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
NAME = "ILSVRC base ours, with softmax"

model_path = os.path.join("model_parameters", "jx_vit_base_p16_224-80ecf9dd.pth")

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



test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


validateset = ImageNetDataset(
    # img_dir="C:\\임새봄\\Dataset\\ILSVRC\\Data\\CLS-LOC\\val",
    img_dir = "/home/NAS_mount/sbim/Dataset/ILSVRC/Data/CLS-LOC/val",
    # annotation_dir= "C:\\임새봄\\Dataset\\ILSVRC\\Annotations\\CLS-LOC\\val",
    annotation_dir = "/home/NAS_mount/sbim/Dataset/ILSVRC/Annotations/CLS-LOC/val",
    transforms=test_transform,
)

validloader = DataLoader(
    dataset = validateset,
    batch_size=batch_size,
    shuffle = False,
)


model = timm_detach.create_model(MODEL, pretrained=True, num_classes=1000).to(device)
model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', map_location=device))
# if args.method =="single":
method = AGCAM(model)



with torch.enable_grad():
    Threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_img = 0
    pixel_acc = [0.0] * len(Threshold)
    dice = [0.0] * len(Threshold)
    precision = [0.0] * len(Threshold)
    recall = [0.0] * len(Threshold)
    iou = [0.0] * len(Threshold)

    for data in tqdm(validloader):
        image = data['image'].cuda()
        label = data['label'].cuda()
        bnd_box = data['bnd_box'].cuda().squeeze(0)

        prediction, mask = method.generate(image)
        if prediction!=label:
            continue
        # print(mask.shape)
        mask = mask.reshape(1, 1, 14, 14)
        upsample = nn.Upsample(224, mode = 'bilinear', align_corners=False)
        mask = upsample(mask)
        mask = (mask-mask.min())/(mask.max()-mask.min())
        seg_label = box_to_seg(bnd_box)
        for i in range(len(Threshold)):

            mask_bnd_box = getBoudingBox_multi(mask, threshold=Threshold[i]).to(device)
            seg_mask = box_to_seg(mask_bnd_box)
            
            
            output = seg_mask.view(-1, )
            target = seg_label.view(-1, ).float()

            tp = torch.sum(output * target)  # TP
            fp = torch.sum(output * (1 - target))  # FP
            fn = torch.sum((1 - output) * target)  # FN
            tn = torch.sum((1 - output) * (1 - target))  # TN
            eps = 1e-5
            pixel_acc_ = (tp + tn + eps) / (tp + tn + fp + fn + eps)
            dice_ = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            precision_ = (tp + eps) / (tp + fp + eps)
            recall_ = (tp + eps) / (tp + fn + eps)
            iou_ = (tp + eps) / (tp + fp + fn + eps)
            print(pixel_acc_)
            pixel_acc[i] += pixel_acc_
            dice[i] += dice_
            precision[i] += precision_
            recall[i] += recall_
            iou[i] += iou_
        num_img+=1



        

print(NAME)
print("number of images: ")
print(num_img)


for i in range(len(Threshold)):
    print("#============================================================================")
    print("Threshold: ", str(Threshold[i]))
    print("pixel_acc: ") 
    print(pixel_acc[i]/num_img)

    print("iou: ")
    print(iou[i]/num_img)

    print("dice: ")
    print(dice[i]/num_img)

    print("precision: ")
    print(precision[i]/num_img)

    print("recall: ")
    print(recall[i]/num_img)




        