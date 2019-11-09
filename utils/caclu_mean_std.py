from torch.utils import data
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
mean=(0.397657144,0.351649219,0.305031406)
class CrowdaiData(data.Dataset):

    def __init__(self,jsonpath,imagedir,seed=2019):

        self.jsonpath=jsonpath
        self.imagedir=imagedir
        self.seed=seed
        self.coco=COCO(self.jsonpath)
        self.cat_ids=self.coco.loadCats(self.coco.getCatIds())
        self.Imageids=self.coco.getImgIds(catIds=self.coco.getCatIds())
        print('Dataset length:  ',len(self.Imageids))
        self.std=np.zeros((3),np.float64)
        self.mean = np.zeros((3), np.float64)
        for bd,index in enumerate(self.Imageids):
            img = self.coco.loadImgs(index)[0]
            imagepath = os.path.join(self.imagedir, img['file_name'])
            image = cv2.imread(imagepath)
            image=np.array(image,np.float64)/255.0
            # image=image-mean
            tstd=np.std(image,axis=(0,1))
            tmean=np.mean(image,axis=(0,1))


            if index%200==0:
                print(index, '  ok:  ',tstd)
            self.std=self.std+tstd
            self.mean=self.mean+tmean
        print(self.std)
        print(self.mean)
        print(len(self.Imageids))
        print(self.std / len(self.Imageids))
        print(self.mean / len(self.Imageids))


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
    # trainjsonpath='/Disk4/xkp/Segmenttion/xkp/Dataset/Crowdai/train/annotation.json'
    # trainimagedir='/Disk4/xkp/Segmenttion/xkp/Dataset/Crowdai/train/images'
    # valjsonpath='/Disk4/xkp/Segmenttion/xkp/Dataset/Crowdai/val/annotation.json'
    # valimagedir='/Disk4/xkp/Segmenttion/xkp/Dataset/Crowdai/val/images'
    #
    # CrowdaiData=CrowdaiData(trainjsonpath,trainimagedir)
    # gen=data.DataLoader(CrowdaiData,1)
    # # for data in gen:
    # #     image,mask=data
    # #     print('mask')
    # [7306.00776323 7893.00046605 8609.13651764]
    # 60317
    # [33998.44831082 36739.30981294 40044.72030112]
    # 280741
    a=[7306.00776323,7893.00046605,8609.13651764]
    a1=60317
    b=[33998.44831082,36739.30981294,40044.72030112]
    b1=280741
    m=[41304.45607405,44632.31027899,48653.85681876]
    n=341058
    print([m[i]/n for i in range(3)])
