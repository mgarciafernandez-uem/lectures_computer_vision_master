import torch
import torchvision
import torch.nn.functional as F
import PIL
import os
import pandas
import math

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from matplotlib import pyplot
from matplotlib import patches


class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_coord = 5
        self.l_noobj = 0.5

    def forward(self, p, a):
        # Calculate IOU of each predicted bbox against the ground truth bbox
        iou = get_iou(p, a)                     # (batch, S, S, B, B)
        max_iou = torch.max(iou, dim=-1)[0]     # (batch, S, S, B)

        # Get masks
        bbox_mask = bbox_attr(a, 4) > 0.0
        p_template = bbox_attr(p, 4) > 0.0
        obj_i = bbox_mask[..., 0:1]         # 1 if grid I has any object at all
        responsible = torch.zeros_like(p_template).scatter_(       # (batch, S, S, B)
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),                # (batch, S, S, B)
            value=1                         # 1 if bounding box is "responsible" for predicting the object
        )
        obj_ij = obj_i * responsible        # 1 if object exists AND bbox is responsible
        noobj_ij = ~obj_ij                  # Otherwise, confidence should be 0

        # XY position losses
        x_losses = mse_loss(
            obj_ij * bbox_attr(p, 0),
            obj_ij * bbox_attr(a, 0)
        )
        y_losses = mse_loss(
            obj_ij * bbox_attr(p, 1),
            obj_ij * bbox_attr(a, 1)
        )
        pos_losses = x_losses + y_losses

        # Bbox dimension losses
        p_width = bbox_attr(p, 2)
        a_width = bbox_attr(a, 2)
        width_losses = mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + config.EPSILON),
            obj_ij * torch.sqrt(a_width)
        )
        p_height = bbox_attr(p, 3)
        a_height = bbox_attr(a, 3)
        height_losses = mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + config.EPSILON),
            obj_ij * torch.sqrt(a_height)
        )
        dim_losses = width_losses + height_losses

        # Confidence losses (target confidence is IOU)
        obj_confidence_losses = mse_loss(
            obj_ij * bbox_attr(p, 4),
            obj_ij * torch.ones_like(max_iou)
        )
        noobj_confidence_losses = mse_loss(
            noobj_ij * bbox_attr(p, 4),
            torch.zeros_like(max_iou)
        )

        # Classification losses
        class_losses = mse_loss(
            obj_i * p[..., :config.C],
            obj_i * a[..., :config.C]
        )

        total = self.l_coord * (pos_losses + dim_losses) \
                + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses
        return total / config.BATCH_SIZE


def mse_loss(a, b):
    flattened_a = torch.flatten(a, end_dim=-2)
    flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(
        flattened_a,
        flattened_b,
        reduction='sum'
    )


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]       # (batch, S, S, B, B)

    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = config.EPSILON
    intersection[zero_unions] = 0.0

    return intersection / union


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::5]


def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)

def to_yolo_bounding_box(raw_dataset, classes, nboxes, grid_layout=(7, 7), pixel_sixe=(448, 448)):

    grid_size_x = int(pixel_sixe[0] / grid_layout[0])
    grid_size_y = int(pixel_sixe[1] / grid_layout[1])

    grid_centers_x = [int((l + 0.5) * grid_size_x) for l in range(grid_layout[0])]
    grid_centers_y = [int((l + 0.5) * grid_size_y) for l in range(grid_layout[1])]

    yolo_dataset = list()

    for image in raw_dataset:

        tensor_element = torch.zeros((1, grid_layout[0], grid_layout[1], 5 * nboxes + len(classes)))

        for element in image:

            class_id = classes.index(element['cell_type'])

            center_x = element['xmin'] + (element['xmax'] - element['xmin']) / 2.
            center_y = element['ymin'] + (element['ymax'] - element['ymin']) / 2.

            grid_element_x = int(center_x / grid_size_x)
            grid_element_y = int(center_y / grid_size_y)

            grid_center_x = center_x - grid_centers_x[grid_element_x]
            grid_center_y = center_y - grid_centers_y[grid_element_y]
            size_x = element['xmax'] - element['xmin']
            size_y = element['ymax'] - element['ymin']


            tensor_element[0][grid_element_x][grid_element_y][0] = 1.
            tensor_element[0][grid_element_x][grid_element_y][1] = grid_center_x
            tensor_element[0][grid_element_x][grid_element_y][2] = grid_center_y
            tensor_element[0][grid_element_x][grid_element_y][3] = size_x
            tensor_element[0][grid_element_x][grid_element_y][4] = size_y
            tensor_element[0][grid_element_x][grid_element_y][5 * nboxes + class_id] = 1

        yolo_dataset.append(tensor_element)

    yolo_dataset = torch.cat(yolo_dataset)


    return yolo_dataset

def from_yolo_bounding_box(yolo_dataset, grid_layout=(7, 7), pixel_sixe=(448, 448)):

    grid_size_x = int(pixel_sixe[0] / grid_layout[0])
    grid_size_y = int(pixel_sixe[1] / grid_layout[1])

    grid_centers_x = [int((l + 0.5) * grid_size_x) for l in range(grid_layout[0])]
    grid_centers_y = [int((l + 0.5) * grid_size_y) for l in range(grid_layout[1])]

    box_dataset = list()
    prob_box_dataset = list()
    prob_class_dataset = list()

    for element in yolo_dataset:

        box_element = torch.zeros(1, config.S * config.S * config.B, 4)
        prob_box_element = torch.zeros(1, config.B * config.S * config.S)
        prob_class_element = torch.zeros(1, config.S * config.S, config.C)

        for grid_x in range(config.S):
            for grid_y in range(config.S):
                for box in range(config.B):
                    cx = element[grid_x][grid_y][5 * box + 1].item()  + grid_centers_x[grid_x]
                    cy = element[grid_x][grid_y][5 * box + 2].item() + grid_centers_y[grid_y]
                    w = element[grid_x][grid_y][5 * box + 3].item()
                    h = element[grid_x][grid_y][5 * box + 4].item()

                    box_element[0][grid_x * config.S + grid_y + config.S * config.S * box][0] = cx
                    box_element[0][grid_x * config.S + grid_y + config.S * config.S * box][1] = cy
                    box_element[0][grid_x * config.S + grid_y + config.S * config.S * box][2] = w
                    box_element[0][grid_x * config.S + grid_y + config.S * config.S * box][3] = h

                    prob_box_element[0][config.S * config.S * box + config.S * grid_x + grid_y] = element[grid_x][grid_y][5 * box]

                for c in range(config.C):
                    prob_class_element[0][config.S * grid_x + grid_y][c] = element[grid_x][grid_y][-3 + c]


        box_dataset.append(box_element)
        prob_class_dataset.append(prob_class_element)
        prob_box_dataset.append(prob_box_element)

    box_dataset = torch.cat(box_dataset)
    prob_class_dataset = torch.cat(prob_class_dataset)
    prob_box_dataset = torch.cat(prob_box_dataset)

    return box_dataset, prob_box_dataset, prob_class_dataset
            


class config:
    S = 7       # Divide each image into a SxS grid
    B = 2       # Number of bounding boxes to predict
    C = 3      # Number of classes in the dataset
    EPSILON = 1E-6
    BATCH_SIZE = 32


#################################
#       Transfer Learning       #
#################################
class YOLOv1ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        # Load backbone ResNet
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)            # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048)              # 4 conv, 2 linear
        )

    def forward(self, x):
        return self.model.forward(x)


class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels):
        super().__init__()

        inner_channels = 1024
        self.depth = 5 * config.B + config.C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channels, 4096),
            # nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, config.S * config.S * self.depth)
        )

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (-1, config.S, config.S, self.depth)
        )


###########################
#       From Scratch      #
###########################
class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.B * 5 + config.C

        layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),                   # Conv 1
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),                           # Conv 2
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),                                     # Conv 3
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(4):                                                          # Conv 4
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(2):                                                          # Conv 5
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        ]

        for _ in range(2):                                                          # Conv 6
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        layers += [
            nn.Flatten(),
            nn.Linear(config.S * config.S * 1024, 4096),                            # Linear 1
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, config.S * config.S * self.depth),                      # Linear 2
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0), config.S, config.S, self.depth)
        )


#############################
#       Helper Modules      #
#############################
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))


if __name__ == '__main__':

    images = os.listdir('./blood_cell/BCCD_Dataset-master/BCCD/')
    image_tensor = list()
    bounding_boxes = list()

    annotations = pandas.read_csv('./blood_cell/BCCD_Dataset-master/annotations.csv')

    for image in images:
        dd = PIL.Image.open(f'./blood_cell/BCCD_Dataset-master/BCCD/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torch.tensor(tt, dtype=torch.float)

        xscaler = 448./tt.shape[2]
        yscaler = 448./tt.shape[1]

        tt = torchvision.transforms.functional.resize(tt, (448, 448))
        tt = tt[None, :, : :]

        image_tensor.append(tt)

        rows = annotations.loc[annotations['filename'] == image]
        rows = rows.reset_index()
        bounding_boxes.append([{'cell_type':rows['cell_type'][cellid],
                                'xmin': int(rows['xmin'][cellid] * xscaler),
                                'xmax': int(rows['xmax'][cellid] * xscaler),
                                'ymin': int(rows['ymin'][cellid] * yscaler), 
                                'ymax': int(rows['ymax'][cellid] * yscaler)} for cellid in range(len(rows))])

    image_tensor = torch.cat(image_tensor)

    yolo_annotations = to_yolo_bounding_box(raw_dataset=bounding_boxes, classes=['WBC', 'RBC', 'Platelets'], nboxes=config.B)


    yolo = YOLOv1ResNet()

    yolo_loss = SumSquaredErrorLoss()
    optim = torch.optim.Adam(yolo.parameters(), lr=1.e-3)

    dataloader_train_image = torch.utils.data.DataLoader(image_tensor, batch_size=config.BATCH_SIZE)
    dataloader_train_target = torch.utils.data.DataLoader(yolo_annotations, batch_size=config.BATCH_SIZE)

    loss_list  = list()

    for epoch in range(20):

        running_loss = 0.
        for image, target in zip(dataloader_train_image, dataloader_train_target):
        
            yolo.train()
            optim.zero_grad()

            pred = yolo(image)
            loss = yolo_loss(pred, target)
            loss.backward()
            optim.step()

            running_loss += loss.item()

            del image, target

        loss_list.append(running_loss)

        print(f'Epoch {epoch} - Loss {running_loss}')


    torch.save(yolo.state_dict(), './blood_cell/yolo_trained.pt')


    pyplot.clf()
    pyplot.plot(loss_list)
    pyplot.yscale('log')
    pyplot.savefig('./blood_cell/loss.png')
