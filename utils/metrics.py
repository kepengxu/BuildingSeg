
import torch
import numpy as np
iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def ioubase(img_true, img_pred):
    img_pred = (img_pred > 0).float()
    i = (img_true * img_pred).sum()
    u = (img_true + img_pred).sum()
    return i / u if u != 0 else u

def iout(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (iou_thresholds <= ioubase(imgs_true[i], imgs_pred[i])).mean()
    return scores.mean()

def IoU(yt,pr):
    num_images = len(yt)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if yt[i].sum() == pr[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i]=ioubase(yt,pr)
    return scores.mean()
