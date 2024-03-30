import torch
import PIL
import torchvision
import pandas
import os
from instance_segmentation import config, from_yolo_bounding_box, YOLOv1ResNet

from matplotlib import patches, pyplot

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

    yolo = YOLOv1ResNet()
    yolo.load_state_dict(torch.load('./blood_cell/yolo_trained.pt'))

    pred = yolo(image_tensor[:10])
    box, prob_box, prob = from_yolo_bounding_box(pred)

    for i in range(len(box)):
        fig, ax = pyplot.subplots()
        ax.imshow(torch.permute(image_tensor[i], (1, 2, 0))/255.)
        for j in range(len(box[i])):
            cx, cy, w, h = box[i][j].detach().numpy()
            print(cx, cy, w, h)
            cx -= w/2.
            cy -= h/2.
            ax.add_patch(patches.Rectangle(xy=(cx, cy), width=w**2, height=h**2, alpha=0.3, color='b'))
            ax.text(cx, cy, f'{prob_box[i][j]:.1f}')

        for square in bounding_boxes[i]:
            cx = square['xmin']
            cy = square['ymin']
            w = (square['xmax'] - square['xmin'])
            h = (square['ymax'] - square['ymin'])
            ax.add_patch(patches.Rectangle(xy=(cx, cy), width=w, height=h, alpha=0.3, color='r'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        fig.savefig(f'./blood_cell/results/image_{i}')