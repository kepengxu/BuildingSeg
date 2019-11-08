
from torch.utils import data
import os
from math import sqrt
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
import os
import random

class CrowdaiData(data.Dataset):
    def __init__(self,datasetdir,augWtarget=True,K=True,shape=160,seed=2019):
        self.datasetdir=datasetdir
        if K:
            self.imagedir=os.path.join(datasetdir,'train','image')
            self.maskdir=os.path.join(datasetdir,'train','label')
            self.listfile=os.path.join(datasetdir,'train','train.txt')

        else:
            self.imagedir = os.path.join(datasetdir, 'val', 'image')
            self.maskdir=os.path.join(datasetdir, "val ", 'label')
            self.listfile = os.path.join(datasetdir, 'train', 'val.txt')
        self.imagelist = []
        with open(self.listfile) as f:
            for line in f:
                self.imagelist.append(line.replace('\n', ''))
        self.shape=shape
        self.seed=seed
        self.augWtarget=True
        self.is_train=K
        self.TR=self.Auger()
        print('Dataset length:  ',len(self.imagelist))
        np.random.seed(seed)
        random.seed(seed)

        self.samplesize = sqrt(2 * self.shape * self.shape)+3
        self.pixnumbers=self.samplesize*self.samplesize*3
        self.kk=np.zeros((self.samplesize,self.samplesize,3)).astype(np.bool)
    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, k):
        filename=self.imagelist[k]
        oimage=cv2.imread(os.path.join(self.imagedir,filename))
        omask=cv2.imread(os.path.join(self.maskdir, filename))
        while True:
            y0=random.randint(0,oimage.shape[0]-self.samplesize-5)
            x0=random.randint(0,oimage.shape[1]-self.samplesize-5)
            image=oimage[y0:y0+self.samplesize,x0:x0+self.samplesize,:]
            mask=omask[y0:y0+self.samplesize,x0:x0+self.samplesize]
            boolimage=image.astype(np.bool)
            t =  1.0*np.sum(self.kk^boolimage)/self.pixnumbers
            if t>0.9:
                # have few zeros pix
                break




  
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
                        rotate_limit=5,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_REFLECT),
                    PadIfNeeded(self.padshape, self.padshape),

                ]
            )
        List_Transforms.extend(
            [

                # Normalize(mean=(0.397657144,0.351649219,0.305031406),std=(0.229, 0.224, 0.225)),
                RandomCrop(self.shape, self.shape),
                ToTensor(),

            ]
        )
        TR=Compose(List_Transforms)
        return TR
