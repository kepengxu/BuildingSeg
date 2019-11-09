
from skimage.segmentation import find_boundaries
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
w0 = 10
sigma = 20

def make_weight_map(masks):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.

    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)

    """
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ
#
# params = [(20, 16, 10), (44, 16, 10), (47, 47, 10)]
# masks = np.zeros((3, 256,256))
# for i, (cx, cy, radius) in enumerate(params):
#     rr, cc = draw.circle(cx, cy, radius)
#     masks[i, rr, cc] = 1


def GetWeights(masks):
    masks = masks[np.newaxis, :, :]
    masks = np.concatenate([masks, masks, masks], 0)
    masks = np.array(masks > 127, np.float64)
    print(masks.shape)
    weights = make_weight_map(masks)
    return weights



import cv2
#
masks=cv2.imread('/Disk4/xkp/project/SegDeepLab/testpicture/35ann35.jpg',cv2.IMREAD_GRAYSCALE)
masks=masks[np.newaxis,:,:]
masks=np.concatenate([masks,masks,masks],0)
masks=np.array(masks>127,np.float64)

print(masks.shape)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
ax1.imshow(masks.sum(0))
ax1.set_axis_off()
ax1.set_title('True Masks', fontsize=15)

weights = make_weight_map(masks)
# weights=cv2.GaussianBlur(weights,(3,3),0)
pos = ax2.imshow(weights)
ax2.set_axis_off()
ax2.set_title('Weights', fontsize=15)
_ = fig.colorbar(pos, ax=ax2)
plt.show()