import cv2
import numpy as np
import matplotlib.pyplot as plt

import gdal
from shapely.geometry import Polygon
from centerline.geometry import Centerline
import math

polygon = Polygon([[0, 0], [0, 4], [4, 4], [4, 0]])
attributes = {"id": 1, "name": "polygon", "valid": True}
centerline = Centerline(polygon, **attributes)
print(centerline.id == 1)

x  = math.log2(3, 2)


"""
def nothing(x):
    pass


img_file = r'I:\DVRPC\Fill_gap\StreetView\images\kXg-PRDBdAix6L11lAtONA_-75.675715_40.050323_0_74.77_landcover.png'

cv2.namedWindow('image')

img = cv2.imread(img_file)
cv2.namedWindow('image')
cv2.createTrackbar('Er/Di', 'image', 0, 1, nothing)
# 创建腐蚀或膨胀选择滚动条，只有两个值
cv2.createTrackbar('size', 'image', 0, 21, nothing)
# 创建卷积核大小滚动条


while (1):
    s = cv2.getTrackbarPos('Er/Di', 'image')
    si = cv2.getTrackbarPos('size', 'image')
    # 分别接收两个滚动条的数据
    k = cv2.waitKey(1)

    kernel = np.ones((si, si), np.uint8)
    # 根据滚动条数据确定卷积核大小
    # erroding = cv2.erode(img, kernel)
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    erroding = cv2.morphologyEx(img, cv2.MORPH_OPEN, g)
    dilation = cv2.dilate(img, kernel)
    # dilation = cv2.morphologyEx(img, cv2.MORPH_CLOSE, 3)
    if k == 27:
        break
    # esc键退出
    if s == 0:
        cv2.imshow('MORPH_OPEN', erroding)
    else:
        cv2.imshow('MORPH_CLOSE', dilation)
"""




