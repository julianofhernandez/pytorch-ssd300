from torchvision import transforms
from utils import (
    rev_label_map, 
)
from PIL import Image, ImageDraw

import torch
import cv2
import numpy as np

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(rev_label_map), 3))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint.
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
print(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms.
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be 
        considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one 
        with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes,
         keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image 
        or you do not want in the image, a list

    Returns: 
        annotated_image: annotated image, a NumPy image.
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output.
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, 
        predicted_scores, 
        min_score=min_score,
        max_overlap=max_overlap, 
        top_k=top_k
    )

    # Move detections to the CPU.
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions.
    original_dims = torch.FloatTensor(
        [
            original_image.width, 
            original_image.height, 
            original_image.width, 
            original_image.height
        ]
        ).unsqueeze(0)
    det_boxes=det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. 
    # ['background'] in SSD300.detect_objects() in model.py.
    if det_labels == ['background']:
        # Just return original image.
        return original_image

    # Annotate.
    annotated_image = original_image.copy()
    annotated_image = np.array(annotated_image, dtype=np.uint8)

    # Suppress specific classes, if needed.
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        
        # Draw boxes.
        box_location = det_boxes[i].tolist()
        cv2.rectangle(
            annotated_image, 
            (int(box_location[0]), int(box_location[1])),
            (int(box_location[2]), int(box_location[3])),
            color=COLORS[det_labels.index(det_labels[i])],
            thickness=2,
            lineType=cv2.LINE_AA
        )
        # Annotate with class label.
        cv2.putText(
            annotated_image, 
            text=det_labels[i],
            org=(int(box_location[0]+1), int(box_location[1]-5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=COLORS[det_labels.index(det_labels[i])],
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return annotated_image[:, :, ::-1]


if __name__ == '__main__':
    img_path = '/mnt/wwn-0x500a0751e6282b63-part2/my_data/Data_Science/Projects/Computer_Vision/object_detection/input/pascal_voc_original/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    result = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
    cv2.imshow('Image', result)
    cv2.waitKey(0)