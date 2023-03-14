import cv2
import numpy as np

image = cv2.imread('../data/images/zidane.jpg')
original_grasp_bboxes = np.array([[[361, 260.582], [301, 315], [320, 336], [380, 281.582]]], dtype=np.int32)
prediction_grasp_bboxes = np.array([[[301, 290.582], [321, 322], [310, 346], [380, 291.582]]], dtype=np.int32)
im = np.zeros(image.shape[:2], dtype="uint8")
im1 = np.zeros(image.shape[:2], dtype="uint8")
original_grasp_mask = cv2.fillPoly(im, original_grasp_bboxes, 255)
prediction_grasp_mask = cv2.fillPoly(im1, prediction_grasp_bboxes, 255)
masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)
masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)

or_area = np.sum(np.float32(np.greater(masked_or, 0)))
and_area = np.sum(np.float32(np.greater(masked_and, 0)))
IOU = and_area / or_area

print(or_area)
print(and_area)
print(IOU)
