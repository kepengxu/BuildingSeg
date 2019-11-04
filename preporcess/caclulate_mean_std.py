from torch.utils import data
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
class CrowdaiData(data.Dataset):
    def __init__(self,jsonpath,imagedir,seed=2019):

        self.jsonpath=jsonpath
        self.imagedir=imagedir
        self.seed=seed
        self.coco=COCO(self.jsonpath)
        self.cat_ids=self.coco.loadCats(self.coco.getCatIds())
        self.Imageids=self.coco.getImgIds(catIds=self.coco.getCatIds())
        print('Dataset length:  ',len(self.Imageids))
        self.mean=np.zeros((3),np.float64)
        for bd,index in enumerate(self.Imageids):
            print(bd, '  ok:  ',index)
            img = self.coco.loadImgs(index)[0]
            imagepath = os.path.join(self.imagedir, img['file_name'])
            image = cv2.imread(imagepath)
            image=np.array(image,np.float64)/255.0
            tmean=np.mean(image,axis=(0,1))
            self.mean=self.mean+tmean

        self.mean=self.mean/float(len(self.Imageids))
        print(self.mean)


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
        MASK = Tmask >= 0.9
        MASK = 1 * MASK
        return [image,MASK]


if __name__=='__main__':
    trainjsonpath='/home/cooper/Dataset/Crowdai/train/annotation-small.json'
    trainimagedir='/home/cooper/Dataset/Crowdai/train/images'
    valjsonpath='/home/cooper/Dataset/Crowdai/val/annotation.json'
    valimagedir='/home/cooper/Dataset/Crowdai/val/images'

    CrowdaiData=CrowdaiData(trainjsonpath,trainimagedir)
    gen=data.DataLoader(CrowdaiData,1)
    # for data in gen:
    #     image,mask=data
    #     print('mask')

#   汇总融合之前的文章，再把ICONIP没注册的那个扩展一下，加数据集实验写SCI
#   NAS 目标检测的NAS
#   基于supernet的NAS

