
import torch
import numpy as np
from segmentation_models_pytorch.utils.metrics import IoUMetric,FscoreMetric

iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

# def ioubase(img_true, img_pred):
#     img_pred = np.array((img_pred > 0.5),np.float)
#     i = (img_true * img_pred).sum()
#     u = (img_true + img_pred).sum()
#     return i / u if u != 0 else u

F=IoUMetric(activation='none')


def StandIoU(yp,yt):
    inter=np.sum(np.logical_and(np.array(yp) > 0.5,np.array(yt)))
    union = np.sum(np.logical_or(np.array(yp) > 0.5, np.array(yt)))
    return inter/(union+0.000001)
ioubase=F

def iout(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (iou_thresholds <= ioubase(imgs_pred[i],imgs_true[i]).numpy()).mean()
    return list(scores)

def IoU(pr,yt):
    num_images = len(yt)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if yt[i].sum() == pr[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i]=ioubase(pr[i],yt[i])
    return list(scores)


def f1():
    F=FscoreMetric(activation='none')
    def F1score(pr,yt):
        return F(pr,yt)
    return F1score
F1score=f1()

def accuracy(preds,label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum