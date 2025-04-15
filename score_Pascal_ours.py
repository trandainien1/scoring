import os
import timm_detach
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gc
from Datasets.PascalVOC import PascalVOC2012Dataset_remove_multi_class
from AGCAM import AGCAM
from tqdm import tqdm
from functions.multi_object_mask import getBoudingBox_multi, box_to_seg
from torch.utils.data import DataLoader
import numpy as np
import random

MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
NAME = "Pascal VOC 2012 ours"

model_path = os.path.join("model_parameters", "pascal_0117_epoch52.pt")

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

root_dir = '/home/NAS_mount/sbim/Dataset/Pascal VOC 2012/trainval/VOCdevkit/VOC2012'
mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

valid_transform = transforms.Compose([transforms.Resize((img_size, img_size)),          
                                      transforms.ToTensor(), 
                                    transforms.Normalize(mean = mean, std = std),
                                      ])

unnormalize = transforms.Compose([
    transforms.Normalize([0., 0., 0.], [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.,])
])


dataset_val = PascalVOC2012Dataset_remove_multi_class(root=root_dir, train=False, transforms=valid_transform)
valid_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)


model = timm_detach.create_model(MODEL, pretrained=True, num_classes=20).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
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

    for data in tqdm(valid_loader):
        image = data['image'].cuda()
        label = data['label'].cuda()
        bnd_box = data['bnd_box'].cuda().squeeze(0)

        prediction, mask = method.generate(image)
        if prediction!=label:
            continue
        # mask = mask.reshape(1, 1, 14, 14)
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




        