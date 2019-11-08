# path='/home/cooper/Dataset/AIRS_DATASET/trainval/val/image/christchurch_172.tif'
# import cv2
# image=cv2.imread(path,cv2.IMREAD_UNCHANGED)
# print(image.shape)
# from libtiff import TIFF
# from osgeo import gdal
# dataset = gdal.Open(path)
#
# print(dataset.GetDescription())#数据描述
# cols=dataset.RasterXSize#图像长度
# rows=(dataset.RasterYSize)#图像宽度
# print(dir(dataset))
# print(dataset.GetMetadata)
# band = dataset.GetRasterBand(1)
# print('Band Type=',gdal.GetDataTypeName(band.DataType))
# data=band.ReadAsArray()
# print('fd')
# f=open('/home/cooper/Dataset/AIRS_DATASET/trainval/train/train.txt')
# for line in f:
#     a=line.replace('\n','')
#     print(a)

import numpy as np

a=np.ones((4,4))
a[1:3,1]=0
a=a.astype(np.bool)

b=np.zeros((4,4),np.bool)

print(a)
print(b)
print(np.sum(a^b)/)