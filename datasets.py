import torch
import os
import cv2
import numpy as np

from torch.utils.data import DataLoader, Dataset
from xml.etree import ElementTree as et
from transforms import get_train_aug, get_train_transform, get_valid_transform

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
        if self.is_train:
            with open(
                os.path.join(data_folder, 'VOC2012', 'ImageSets', 'Main', 'trainval.txt'), 'r'
            ) as f:
                self.image_names_07.extend(f.readlines())
                # Populate the proper image paths into the 
                # `image_paths` list.
                for name in self.image_names_07:
                    name = name.strip('\n')
                    self.image_paths.append(os.path.join(
                        data_folder, 'VOC2012', 'JPEGImages', name+'.jpg'
                    ))
            with open(
                os.path.join(data_folder, 'VOC2007', 'ImageSets', 'Main', 'trainval.txt'), 'r'
            ) as f:
                self.image_names_12.extend(f.readlines()) 
                # Populate the proper image paths into the 
                # `image_paths` list.
                for name in self.image_names_12:
                    name = name.strip('\n')
                    self.image_paths.append(os.path.join(
                        data_folder, 'VOC2007', 'JPEGImages', name+'.jpg'
                    ))
            self.image_names.extend(self.image_names_07)
            self.image_names.extend(self.image_names_12)
        else:
            with open(
                os.path.join(data_folder, 'VOC2007', 'ImageSets', 'Main', 'test.txt'), 'r'
            ) as f:
                self.image_names.extend(f.readlines()) 
                # Populate the proper image paths into the 
                # `image_paths` list.
                for name in self.image_names:
                    name = name.strip('\n')
                    self.image_paths.append(os.path.join(
                        data_folder, 'VOC2007', 'JPEGImages', name+'.jpg'
                    ))

    def load_image_and_labels(self, index):
        image_name = self.image_names[index]
        image_name = image_name.strip('\n')
        image_path = self.image_paths[index]
        # Get either `VOC2007` or `VOC2012`
        year_dir = image_path.split(os.path.sep)[-3]

        image = cv2.imread(image_path)
        # Convert BGR to RGB color format and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))

        # Capture the corresponding XML file for getting the annotations.
        annot_file_path = os.path.join(
            self.data_folder, year_dir, 'Annotations', image_name+'.xml'
        )

        boxes = []
        orig_boxes = []
        labels = []
        difficultues = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # Get the original height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Box coordinates for xml files are extracted and corrected for image size given.
        for member in root.findall('object'):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list. +1 at the end as
            # `background` will take indenx 0.
            labels.append(self.classes.index(member.find('name').text)+1)
            difficultues.append(int(member.find('difficult').text))
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            # Bounding box coordinates without being resized.
            orig_boxes.append([xmin, ymin, xmax, ymax])

            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = ((xmin/image_width)*self.width) / self.width
            xmax_final = ((xmax/image_width)*self.width) / self.width
            ymin_final = ((ymin/image_height)*self.height) / self.height
            ymax_final = ((ymax/image_height)*self.height) /self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

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
        return len(self.image_paths)

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