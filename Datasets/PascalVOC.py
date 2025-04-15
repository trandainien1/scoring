import os
import torch
from glob import glob
import PIL
from torchvision.datasets import ImageFolder
from bs4 import BeautifulSoup

class PascalVOC2012Dataset_remove_multi_class(ImageFolder):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.img_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")
        self.image_sets = os.path.join(root, "ImageSets", "Main")
        self.classes = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
        )
        self.class_num = [0] * 20
        self.transforms = transforms
        self.img_data = []
        self.img_labels = []
        if train==True:
            self.target='train'
            # class_file = os.path.join(self.image_sets, "train_classes.txt")
            class_file = os.path.join(self.image_sets, "train.txt")
        elif train==False:
            self.target='val'
            # class_file = os.path.join(self.image_sets, "val_classes.txt")
            class_file = os.path.join(self.image_sets, "val.txt")

        class_file = open(class_file, "r")
        list = class_file.read().split('\n')
        for line in list:
            data = line.split()
            print('DEBUG: data')
            img_name = data[0]
            labels = data[1:]
            # print(labels.count(1))
            if labels.count('1')==1:
                self.img_data.append(img_name)
                self.img_labels.append(labels.index('1'))
                self.class_num[labels.index('1')]+=1


    def __getitem__(self, idx):
        img_name, label = self.img_data[idx], self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name+".jpg")        
        img = PIL.Image.open(img_path)
        width, height = img.size
        anno_path = os.path.join(self.annotation_dir, img_name+".xml")
        with open(anno_path, 'r') as f:
            file = f.read()
        soup = BeautifulSoup(file, 'html.parser')
        if self.transforms:
            img = self.transforms(img)
            # if img.shape[0]==1:
            #     img = torch.cat((img, img, img), dim=0)
        objects = soup.findAll('object')
        
        bnd_box = torch.tensor([])

        for object in objects:
            class_name = object.text.split('\n')[1]
            if self.classes.index(class_name)==self.img_labels[idx]:
                xmin = int(object.bndbox.xmin.text)
                ymin = int(object.bndbox.ymin.text)
                xmax = int(object.bndbox.xmax.text)
                ymax = int(object.bndbox.ymax.text)
                xmin = int(xmin/width*224)
                ymin = int(ymin/height*224)
                xmax = int(xmax/width*224)
                ymax = int(ymax/height*224)
                if bnd_box.dim()==1:
                    bnd_box = torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)
                else:
                    bnd_box = torch.cat((bnd_box, torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)), dim=0)
            # else :
                # print("warning: mutliple labeled image may be included")
        sample = {'image': img, 'label': label, 'filename': img_name, 'num_objects': len(objects), 'bnd_box': bnd_box, 'img_path': img_path}
        # sample = {'image': img, 'label': label}
        return sample

    def __len__(self):
        return len(self.img_data)

    def get_num_per_class(self) :
        return self.class_num

