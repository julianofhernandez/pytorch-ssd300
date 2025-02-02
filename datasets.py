import torch
import os
import cv2
import numpy as np

from torch.utils.data import DataLoader, Dataset
from xml.etree import ElementTree as et
from transforms import get_train_aug, get_train_transform, get_valid_transform

import json
import subprocess
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib.patches import Rectangle

import random
from PIL import Image
import matplotlib.pyplot as plt


class TACO(Dataset):
    """
    A Custom PyTorch Dataset class to load TACO dataset.
    """

    def __init__(
        self, 
        data_folder='TACO', 
        train=False, 
        keep_difficult=False,
        width=300,
        height=300,
        use_train_aug=False,
        transforms=None,
        classes=None
    ):
        """
        :param data_folder: Path to the TACO repository folder.
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
        self.image_names = [] 
        self.root_dir = os.path.join(data_folder, 'data')

        self.download(data_folder)
        self.split(data_folder)
        
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

    def download(self, dataset_dir):
        dataset_url = "https://github.com/pedropro/TACO.git"
        if not os.path.exists(dataset_dir):
            os.system(f"git clone {dataset_url} {dataset_dir}")
        download_script = os.path.join(dataset_dir, "download.py")
        annotations_file = os.path.join(dataset_dir, "data", "annotations.json")
        try:
            subprocess.run(["python", download_script, '--dataset_path', annotations_file], check=True)
        except subprocess.CalledProcessError as e:
            if e.returncode != 2:
                raise Exception("Download script returned a returncode that's not expected")
        download_check = os.path.join(dataset_dir, 'data', 'batch_1', '000001.jpg')
        if not os.path.exists(download_check):
            raise Exception(f"TACO download failed, try rerunning {download_script}")

    def split(self, dataset_dir):        
        # Split training and validation datasets
        split_script = os.path.join(dataset_dir, "detector", "split_dataset.py")
        subprocess.run(["python", split_script, "--dataset_dir", os.path.join(dataset_dir, "data")], check=True)
        split_check = os.path.join(dataset_dir, 'data', 'annotations_0_train.json')
        if not os.path.exists(split_check):
            raise Exception(f"Split failed, try rerunning {split_check}")

    def round_0_to_1(self, val):
        if val >= 0 and val <= 1:
            return val
        elif val < 0:
            return 0
        elif val > 1:
            return 1
        else:
            return 696969696969

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

        # Get the bounding boxes and categories for each object in the image
        orig_boxes = [ann['bbox'] for ann in annotations]
        #! To narrow the categories later add some extra code here
        labels = [ann['category_id']+1 for ann in annotations]

        # Convert the bbox format from [x, y, width, height] to [x1, y1, x2, y2]
        orig_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in orig_boxes]
        
        boxes = []
        for box in orig_boxes:
            xmin_final = ((box[0]/image_width)*self.width) / self.width
            xmax_final = ((box[2]/image_width)*self.width) / self.width
            ymin_final = ((box[1]/image_height)*self.height) / self.height
            ymax_final = ((box[3]/image_height)*self.height) /self.height

            xmax_final = self.round_0_to_1(xmax_final)
            ymin_final = self.round_0_to_1(ymin_final)
            ymax_final = self.round_0_to_1(ymax_final)
            xmin_final = self.round_0_to_1(xmin_final)

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        difficultues = 1

        return image, image_resized, orig_boxes, boxes, labels, difficultues
    
    def __getitem__(self, idx):
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
    train_dataset = TACO(
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
    valid_dataset = TACO(
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