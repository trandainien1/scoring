
import os
import PIL
import torch
from torchvision.datasets import ImageFolder

class CUBDataset(ImageFolder):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        img_file = os.path.join(root, "images.txt")
        label_file = os.path.join(root, "image_class_labels.txt")
        split_file = os.path.join(root, "train_test_split.txt")
        class_file = os.path.join(root, "classes.txt")
        bnd_box_file = os.path.join(root, "bounding_boxes.txt")

        self.transforms = transforms
        self.img_data = []
        self.img_labels = []
        self.class_names = []
        self.bnd_boxes = []

        file = open(class_file, "r")
        lines = file.read().split('\n')
        self.classes = []
        for line in lines:
            idx, name = line.split(' ')
            self.classes.append(name)  
        file.close()
        split_file = open(split_file, "r")
        lines = split_file.read().split('\n')
        split_list = []
        for line in lines:
            idx, sets = line.split()
            split_list.append(sets)
        split_file.close()
        label_file = open(label_file, "r")
        lines = label_file.read().split('\n')
        label_list = []
        for line in lines:
            idx, label = line.split()
            label_list.append(label)
        label_file.close()

        img_file = open(img_file, "r")
        lines = img_file.read().split('\n')
        img_list = []
        for line in lines:
            idx, img = line.split()
            img_list.append(img)
        img_file.close()

        bnd_box_file = open(bnd_box_file, "r")
        lines = bnd_box_file.read().split('\n')
        bnd_boxes = []
        for line in lines:
            idx, x, y, width, height = line.split()
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
            bnd_boxes.append([x, y, x+width, y+height])
        bnd_box_file.close()
        

        if train==True:
            for i in range(len(img_list)):
                if split_list[i]=='1':
                    self.img_data.append(img_list[i])
                    self.img_labels.append(int(label_list[i])-1)
                    self.bnd_boxes.append(bnd_boxes[i])
        else:
            for i in range(len(img_list)):
                if split_list[i]=='0':
                    self.img_data.append(img_list[i])
                    self.img_labels.append(int(label_list[i])-1)
                    self.bnd_boxes.append(bnd_boxes[i])
        
    def __getitem__(self, idx) :
        img_dir = os.path.join(self.root, "images")
        img_path, label = self.img_data[idx], self.img_labels[idx]
        bnd_box = self.bnd_boxes[idx]
        img_path = os.path.join(img_dir, img_path)
        img = PIL.Image.open(img_path).convert("RGB")
        img_name = img_path.split('/')[-1].split('.')[0]
        width, height = img.size
        if self.transforms:
            img = self.transforms(img)
        xmin, ymin, xmax, ymax = bnd_box
        xmin = int(xmin/width*224)
        ymin = int(ymin/height*224)
        xmax = int(xmax/width*224)
        ymax = int(ymax/height*224)

        bnd_box = torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)
        sample = {'image': img, 'label': label, 'bnd_box': bnd_box, 'filename': img_name}
        return sample
    

    def __len__(self):
        return len(self.img_data)