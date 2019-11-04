from torch.utils import data
import os
import cv2
from preporcess.ContrastEnhance import ContrastEnhancement
import numpy as np
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from pycocotools import mask as cocomask
from imgaug import augmenters as iaa
from preporcess.augmentation import CenterCrop,\
                                    ShiftScaleRotate,\
                                    HorizontalFlip,\
                                    GaussNoise
from albumentations import HorizontalFlip,\
                            VerticalFlip,\
                            Resize,\
                            Normalize, \
                            HueSaturationValue,\
                            PadIfNeeded,\
                            ToFloat,\
                            Compose,\
                            RandomCrop
from albumentations.torch import ToTensor

import random


class CrowdaiData(data.Dataset):
    def __init__(self,jsonpath,imagedir,augWtarget=True,padshape=400,K=True,shape=160,seed=2019):
        self.jsonpath=jsonpath
        self.imagedir=imagedir
        self.shape=shape
        self.seed=seed
        self.coco=COCO(self.jsonpath)
        self.cat_ids=self.coco.loadCats(self.coco.getCatIds())
        self.Imageids=self.coco.getImgIds(catIds=self.coco.getCatIds())
        self.augWtarget=True
        self.is_train=K
        self.padshape=padshape
        self.TR=self.Auger()
        print('Dataset length:  ',len(self.Imageids))
        np.random.seed(seed)
        random.seed(seed)

    def __len__(self):
        return len(self.Imageids)

    def __getitem__(self, k):
        index=self.Imageids[k]
        img=self.coco.loadImgs(index)[0]
        imagepath=os.path.join(self.imagedir,img['file_name'])
        image=cv2.imread(imagepath)
        annid=self.coco.getAnnIds(imgIds=img['id'])
        ann=self.coco.loadAnns(annid)
        Tmask=np.zeros(shape=(300,300))
        for t in ann:
            annseg = t['segmentation']
            rle = cocomask.frPyObjects(annseg, img['height'], img['width'])
            mask = cocomask.decode(rle)
            mask = mask
            mask = mask.reshape(img['height'], img['width'])
            Tmask = Tmask + mask
        image=ContrastEnhancement(image)
        mask = (Tmask >= 0.8).astype('float32')
        augment=self.TR(image=image,mask=mask)
        image=augment['image']
        mask=augment['mask']

        return [image,mask]
    def Auger(self):
        List_Transforms=[]
        if self.is_train:
            List_Transforms.extend(
                [
                    HueSaturationValue(10, 10, 10, p=0.3),

                    HorizontalFlip(0.3),
                    VerticalFlip(p=0.3),

                    # May be it not work,will rescale [0,255]  ->   [0.0,1.0]
                    ToFloat(always_apply=True),


                    ShiftScaleRotate(
                        shift_limit=0.1,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_REFLECT),
                    PadIfNeeded(self.padshape, self.padshape),

                ]
            )
            List_Transforms.extend(
                [
                    RandomCrop(self.shape,self.shape),
                    Normalize(mean=(0.397657144,0.351649219,0.305031406),std=(0.229, 0.224, 0.225)),
                    ToTensor(),
                ]
            )
        TR=Compose(List_Transforms)
        return TR



def GetDataloader(trainimagepath,
                  trainmaskpath,
                  valimagepath,
                  valmaskpath,
                  shape=256,
                  padshape=312,
                  batchsize=16,
                  numworkers=4):
    traindata=CrowdaiData(trainmaskpath,trainimagepath,padshape=padshape,shape=shape)
    trainloader=data.DataLoader(traindata,batch_size=batchsize,shuffle = True, num_workers =numworkers)

    valdata=CrowdaiData(valmaskpath,valimagepath,padshape=padshape,shape=shape)
    valloader=data.DataLoader(valdata,batch_size=batchsize,shuffle = False, num_workers =numworkers)
    return trainloader,valloader



if __name__=='__main__':
    valjsonpath='/home/cooper/Dataset/Crowdai/train/annotation-small.json'
    imagedir='/home/cooper/Dataset/Crowdai/train/images'

    SIZE=300
    # CrowdaiData=CrowdaiData(valjsonpath,imagedir,padshape=312,)
    # gen=data.DataLoader(CrowdaiData,batch_size=1,)
    gen,_=GetDataloader(imagedir,valjsonpath,imagedir,valjsonpath,numworkers=2)



    for bd,data in enumerate(gen):
        image,mask=data
        ndimage,ndmask=image.numpy(),mask.numpy()
        print(ndimage.shape,ndmask.shape,ndimage.max(),ndmask.max())
        print('HHHHHHHHHHHH')
        if len(ndimage.shape)==4 and ndimage.shape[1]==3:
            ndimage = np.moveaxis(ndimage[0,:,:,:],0,-1)
            ndmask=ndmask[0,0,]
            print(' To tensor ')
        else:
        # This code is prepare for visualize
            ndimage=ndimage[0,:,:,:]#.reshape(SIZE,SIZE,3)
            ndmask=ndmask[0,:,:]#.reshape(SIZE,SIZE)
            print(' Not To tensor ')
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(ndimage,cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.imshow(ndmask)
        plt.show()
        print('j')



#   汇总融合之前的文章，再把ICONIP没注册的那个扩展一下，加数据集实验写SCI
#   NAS 目标检测的NAS
#   基于supernet的NAS

