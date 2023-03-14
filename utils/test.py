import cv2
import numpy as np
from shapely.geometry import Polygon
import torch


# 中心点 矩形的w h, 旋转的theta（角度，不是弧度）
def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[0][2] * boxes1[0][3]
    area2 = boxes2[0][2] * boxes2[0][3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1 + area2 - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)


def intersection(g, p):
    # g = np.asarray(g)
    # p = np.asarray(p)
    g = Polygon(g[0])
    p = Polygon(p[0])
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    print(inter)
    union = g.area + p.area - inter
    print(union)
    if union == 0:
        return 0
    else:
        return inter / union


# box1 = np.array([[[361, 260.582], [301, 315], [320, 336], [380, 281.582]]])
# box2 = np.array([[[301, 290.582], [321, 322], [310, 346], [380, 291.582]]])
# box1 = np.array([[[0, 0], [1, 0], [1, 2], [0, 2]], [[0, 0], [1, 0], [1, 2], [0, 2]]])
# box2 = np.array([[[0, 1], [1, 0], [2, 1], [1, 2]], [[0, 0], [1, 0], [1, 2], [0, 2]]])
# box1 = torch.from_numpy(box1)
# box2 = torch.from_numpy(box2)
# box1 = torch.tensor([[[0, 0], [1, 0], [1, 2], [0, 2]], [[0, 0], [1, 0], [1, 2], [0, 2]]])
# box2 = torch.tensor([[[0, 1], [1, 0], [2, 1], [1, 2]], [[0, 0], [1, 0], [1, 2], [0, 2]]])
# aa = torch.zeros(5)
# print(aa)
# print(intersection(box1, box2))

a = torch.tensor([0.1666, 0.5, 0.8333])
print(a.repeat(1, 9).view(3, 3, 3, 1))
