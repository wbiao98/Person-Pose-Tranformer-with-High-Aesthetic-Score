import numpy as np
from scipy.ndimage.filters import gaussian_filter
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
from collections import defaultdict
import sys
from PIL import Image, ImageDraw, ImageColor
import numpy as np
from util.keypoint_model import OPENPOSE_18
import imageio

LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS2 = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
COLORS = [(255, 0, 0, 0),
          (0, 255, 0, 0),
          (0, 0, 255, 0)]

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

def cords_to_map(cords, img_size, old_size=None, affine_matrix=None, sigma=6):
    old_size = img_size if old_size is None else old_size
    cords = cords.astype(float)
    result = np.zeros(img_size+cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        point[0] = point[0] / old_size[0] * img_size[0]
        point[1] = point[1] / old_size[1] * img_size[1]
        if affine_matrix is not None:
            point_ = np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3, 1))
            point_0 = int(point_[1])
            point_1 = int(point_[0])
        else:
            point_0 = int(point[0])
            point_1 = int(point[1])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result

def load_pose_cords_from_strings(y_str,x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def tesor2im(image_tensor,bytes=255.0,imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().detach().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().detach().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)

def kp2stick(kps,size=[512,512],kp_model=OPENPOSE_18):
    # Create canvas
    im = Image.fromarray(np.zeros(size+[3] , dtype='uint8'))
    draw = ImageDraw.Draw(im)

    #change X Y
    size = kps.shape[0]
    for idx in range(size):
        k = kps[idx]
        k1 = k[0]*2
        k[0] = k[1]*2
        k[1] = k1
    # Draw Body Polygon
    body = []
    for idx in kp_model.CENTER_BODY:
        point = kps[idx].tolist()
        if point[0] <= 0 and point[1] <= 0:
            continue

        body += [tuple(point)]
    draw.polygon(body, fill=COLORS[1])

    # Draw Lines
    all_lines = [
        kp_model.LEFT_LINES,
        kp_model.CENTER_LINES,
        kp_model.RIGHT_LINES
    ]
    for channel, lines in enumerate(all_lines):
        for p1idx, p2idx in lines:
            point1 = tuple(list(kps[p1idx]))
            point2 = tuple(list(kps[p2idx]))

            if (point1[0] <= 0 and point1[1] <= 0) \
                    or (point2[0] <= 0 and point2[1] <= 0):
                continue

            draw.line([point1, point2], fill=COLORS[channel], width=4)

    # Draw Points
    point_size = 3
    all_points = [
        kp_model.LEFT_POINTS,
        kp_model.CENTER_POINTS,
        kp_model.RIGHT_POINTS,
    ]
    for channel, points in enumerate(all_points):
        for pidx in points:
            point = kps[pidx]
            if (point[0] <= 0 and point[1] <= 0):
                continue
            box = list(point - point_size) + list(point + point_size)

            draw.ellipse(box, fill=COLORS[channel])

    del draw

    return np.array(im)

