import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import glob
import gluoncv
from skimage import io

from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

import os

import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import gluoncv
import numpy
from PIL import Image

GPU_NUM = 0

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'
class Seg():
    def __init__(self, gpu_num=0):

        self.model = gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True, ctx=mx.gpu(GPU_NUM))
        self.ctx = mx.gpu(gpu_num)

    def getSeg(self, np_img, saved_name=''):
        img = test_transform(np_img, self.ctx)
        # print(img)
        output = self.model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        predict = predict.astype(numpy.uint8)

        if saved_name != '':
            try:
                io.imsave(saved_name, predict)
            except Exception as e:
                print("Error in saving file in getSeg():", e, saved_name)
        return predict

    def getColor(self, predict, dataset='ade20k', saved_name=''):
        colored = get_color_pallete(predict, dataset)
        if saved_name != '':
            try:
                colored.save(saved_name)
            except Exception as e:
                print("Error in saving file in getSeg():", e, saved_name)
        return colored


if __name__ == '__main__':
    seg = Seg()
    filename = r'X:\Shared drives\sidewalk\Street_trees\Philly\tree_jpg\_-_elg9jYBI22BgLxPW4Xw_-75.148922_39.965694_0_173.jpg'
    img = image.imread(filename)
    predict = seg.getSeg(img)
    print(predict)
    colored = seg.getColor(predict)
    print(colored)
    plt.imshow(colored)
    plt.show()
