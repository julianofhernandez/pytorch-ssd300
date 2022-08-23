"""
A lot of utility functions have been borrowed/taken/adapted from:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

Some functions have been renamed to better represent what they do.
"""

import torch
import random
import torchvision.transforms.functional as FT

def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d,
                index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long()
            )

    return tensor

def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h)
     to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, 
        a tensor of size (n_boxes, 4)

    Returns: 
        bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat(
        [cxcy[:, :2] - (cxcy[:, 2:] / 2),     # x_min, y_min
         cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1  # x_max, y_max
    )  

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, 
    since they are encoded in `cxcy_to_xy` function.

    They are decoded into center-size coordinates.

    This is the inverse of the `cxcy_to_xy` function.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, 
        a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, 
        a tensor of size (n_priors, 4)

    Returns: 
        decoded bounding boxes in center-size form, a tensor of 
        size (n_priors, 4)
    """
    return torch.cat(
        [gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
        torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1           # w, h
    ) 

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) 
    w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box,
    and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, 
    and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, 
        a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the 
        encoding must be performed, a tensor of size (n_priors, 4)
    Returns: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in 
    # the original Caffe repo, completely empirical.
    # They are for some sort of numerical conditioning, 
    # for 'scaling the localization gradient'.
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat(
        [(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
          torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1             # g_w, g_h
    )  

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates 
    (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, 
        a tensor of size (n_boxes, 4)
    Returns: bounding boxes in center-size coordinates, 
        a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)    # w, h

def intersection_over_union(
    boxes_preds, boxes_labels, 
    epsilon=1e-6
):
    """
    Calculates intersection over union for bounding boxes.
    
    :param boxes_preds (tensor): Bounding box predictions of shape (BATCH_SIZE, 4)
    :param boxes_labels (tensor): Ground truth bounding box of shape (BATCH_SIZE, 4)
    :param epsilon: Small value to prevent division by zero.
    Returns:
        tensor: Intersection over union for all examples
    """

    lower_bounds = torch.max(boxes_preds[:, :2].unsqueeze(1), boxes_labels[:, :2].unsqueeze(0))
    upper_bounds = torch.min(boxes_preds[:, 2:].unsqueeze(1), boxes_labels[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    intersection =  intersection_dims[:, :, 0] * intersection_dims[:, :, 1]

    box1_area = abs(
        boxes_preds[:, 2] - boxes_preds[:, 0]) * \
        (boxes_preds[:, 3] - boxes_preds[:, 1]
    )
    box2_area = abs(
        boxes_labels[:, 2] - boxes_labels[:, 0]) * \
        (boxes_labels[:, 3] - boxes_labels[:, 1]
    )

    union = (box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - intersection + epsilon)

    return intersection / union




#########################
# FOLLOWING ARE THE ONES WHICH WILL BE CHANGED WITH OWN CODE.
#########################
# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
#########################################################