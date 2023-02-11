import torch
import os
import cv2
import numpy as np

from torch.utils.data import DataLoader, Dataset
from xml.etree import ElementTree as et
from transforms import get_train_aug, get_train_transform, get_valid_transform

import json
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib.patches import Rectangle

import random
from PIL import Image
import matplotlib.pyplot as plt

def checkBetween1and0(bboxValue):
    if bboxValue < 1 and bboxValue > 0:
        return bboxValue
    if bboxValue > 1:
        return 1
    elif bboxValue < 0:
        return 0

class PascalVOCDataset(Dataset):
    """
    A Custom PyTorch Dataset class to load Pascal VOC dataset.
    """

    def __init__(
        self, 
        data_folder='VOCdevkit', 
        train=False, 
        keep_difficult=False,
        width=300,
        height=300,
        use_train_aug=False,
        transforms=None,
        classes=None
    ):
        """
        :param data_folder: Path to the `VOCdevkit` folder.
        :param train: Boolean, wheter to prepare data for training set. If 
            False, then prepare for validation set. The augmentations will be 
            applied accordingly.
        :param keep_difficult: Keep or discard the objects that are marked as 
            difficult in the XML file.
        :param width: Width to reize to.
        :param height: Height to resize to.
        :param use_train_aug: Boolean, whether to apply training augmentation or not.
        :param transforms: Which transforms to apply, training or validation transforms.
            if `use_train_aug` is True, for training set, simple transforms is not applied.
        :param classes = List or tuple containing the class names.
        """
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        self.height = height
        self.width = width
        self.is_train = train
        self.use_train_aug = use_train_aug
        self.transforms = transforms
        self.classes = classes

        self.image_paths = [] # Image to store proper image paths with extension.
        self.image_names_07 = [] # List to store image names for VOC 2007.
        self.image_names_12 = [] # List to store image names for VOC 2012.
        self.image_names = [] 
        self.root_dir = data_folder
        
        # Open annotations file
        if self.is_train:
            with open(os.path.join(self.root_dir, 'annotations_0_train.json'), 'r') as f:
                self.annotations = json.load(f)
        else:
            with open(os.path.join(self.root_dir, 'annotations_0_val.json'), 'r') as f:
                self.annotations = json.load(f)
        # Create images list
        self.image_ids = [image['id'] for image in self.annotations['images']]
        self.image_filepaths = [os.path.join(self.root_dir, image['file_name']) for image in self.annotations['images']]
        self.width_height = [(image['width'], image['height']) for image in self.annotations['images']]
        print(self.width_height)
        # Create classes list
        self.classes = [category['name'] for category in self.annotations['categories']]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def load_image_and_labels(self, index):
        image_path = self.image_filepaths[index]
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))

        # Extract the corresponding annotations
        annotations = [ann for ann in self.annotations['annotations'] if int(ann['image_id']) == index]

        image_width = image.shape[1]
        image_height = image.shape[0]

        boxes = []
        orig_boxes = []
        labels = []

        # Get the bounding boxes and categories for each object in the image
        [orig_boxes.append(ann['bbox']) for ann in annotations]
        #! To narrow the categories later add some extra code here
        [labels.append(ann['category_id']+1) for ann in annotations]

        # Convert the bbox format from [x, y, width, height] to [x1, y1, x2, y2]
        orig_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in orig_boxes]
        
        boxes = []
        # Convert BBOX values to be between 0 and 1
        for box in orig_boxes:
            xmin_final = ((box[0]/image_width)*self.width) / self.width
            xmax_final = ((box[2]/image_width)*self.width) / self.width
            ymin_final = ((box[1]/image_height)*self.height) /self.height
            ymax_final = ((box[3]/image_height)*self.height) /self.height

            xmin_final = checkBetween1and0(xmin_final)
            xmax_final = checkBetween1and0(xmax_final)
            ymin_final = checkBetween1and0(ymin_final)
            ymax_final = checkBetween1and0(ymax_final)

            if xmin_final < 0.0 or xmin_final > 1.0 or \
                ymin_final < 0.0 or ymin_final > 1.0 or \
                xmax_final < 0.0 or xmax_final > 1.0 or \
                ymax_final < 0.0 or ymax_final > 1.0:
                raise ValueError(f"Bounding box values for sample at index {index} are outside of the expected\n \
                range [0.0, 1.0]: ({xmin_final}, {ymin_final}, {xmax_final}, {ymax_final})\n\
                Width: {image_width}; height: {image_height}\
                Original BBOX: ({box[0]}, {box[1]}, {box[2]}, {box[3]})")
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        difficultues = 1

        return image, image_resized, orig_boxes, boxes, labels, difficultues
    
    def __getitem__(self, idx):
        print("image index: " + str(idx))
        image, image_resized, orig_boxes, boxes, labels, diffculties = \
            self.load_image_and_labels(index=idx)

        if self.use_train_aug:
            
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                                     bboxes=boxes,
                                     labels=labels)
            image_resized = sample['image']
            boxes = torch.Tensor(sample['bboxes'])
            labels = torch.Tensor(sample['labels'])
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=boxes,
                                     labels=labels)
            image_resized = sample['image']
            boxes = torch.Tensor(sample['bboxes'])
            labels = torch.tensor(sample['labels'])

        diffculties = torch.ByteTensor(diffculties)
        return image_resized, boxes, labels, diffculties

    def __len__(self):
        return len(self.image_ids)

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.

    :param batch: an iterable of N sets from __getitem__()

    Returns: 
        a tensor of images, lists of varying-size tensors of 
        bounding boxes, labels, and difficulties
    """

    images = list()
    boxes = list()
    labels = list()
    difficulties = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        difficulties.append(b[3])

    images = torch.stack(images, dim=0)
    return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

TACOPath = 'C:\\Users\\julian\\repos\\TACO\\data'

# Prepare the final datasets and data loaders.
def create_train_dataset(
    data_folder, 
    train=True,
    keep_difficult=True,
    resize_width=300, 
    resize_height=300, 
    use_train_aug=False,
    classes=None
):
    train_dataset = PascalVOCDataset(
        data_folder,
        train=train,
        keep_difficult=keep_difficult,
        width=resize_width,
        height=resize_height,
        use_train_aug=use_train_aug,
        transforms=get_train_transform(),
        classes=classes
    )
    return train_dataset

def create_valid_dataset(
    data_folder, 
    train=False,
    keep_difficult=True,
    resize_width=300, 
    resize_height=300, 
    use_train_aug=False,
    classes=None
):
    valid_dataset = PascalVOCDataset(
        data_folder,
        train=train,
        keep_difficult=keep_difficult,
        width=resize_width,
        height=resize_height,
        use_train_aug=use_train_aug,
        transforms=get_valid_transform(),
        classes=classes
    )
    return valid_dataset

def create_train_loader(train_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader

# # Prepare the final datasets and data loaders.
# def create_train_dataset(
#     data_folder, 
#     train=True,
#     keep_difficult=True,
#     resize_width=300, 
#     resize_height=300, 
#     use_train_aug=False,
#     classes=None
# ):
#     train_dataset = TACODataset(TACOPath, os.path.join(TACOPath,'annotations.json'), get_valid_transform())
#     return train_dataset

# def create_valid_dataset(
#     data_folder, 
#     train=False,
#     keep_difficult=True,
#     resize_width=300, 
#     resize_height=300, 
#     use_train_aug=False,
#     classes=None
# ):
#     valid_dataset = TACODataset(TACOPath, os.path.join(TACOPath,'annotations.json'), get_valid_transform())
#     return valid_dataset

# def create_train_loader(train_dataset, batch_size, num_workers=0):
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         collate_fn=collate_fn
#     )
#     return train_loader
# def create_valid_loader(valid_dataset, batch_size, num_workers=0):
#     valid_loader = DataLoader(
#         valid_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         collate_fn=collate_fn
#     )
#     return valid_loader