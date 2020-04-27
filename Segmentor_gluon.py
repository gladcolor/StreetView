import mxnet as mx
import numpy as np
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
import glob
import gluoncv
from skimage import io
from tqdm import tqdm

from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

import os

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
        np_img = np.array(np_img)
        np_img = mx.ndarray.array(np_img)
        img = test_transform(np_img, self.ctx)
        # print(img)
        output = self.model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        predict = predict.astype(numpy.uint8)

        if saved_name != '':
            try:
                io.imsave(saved_name, predict, check_contrast=False)
            except Exception as e:
                print("Error in saving file in getSeg():", e, saved_name)
        return predict

    def getColor(self, predict, dataset='ade20k', saved_name=''):
        colored = get_color_pallete(predict, dataset)
        if saved_name != '':
            try:
                colored.save(saved_name)
                # io.imsave(saved_name, colored, check_contrast=False)
            except Exception as e:
                print("Error in saving file in getSeg():", e, saved_name)
        return colored


def seg_files(folder, seg_path='', color_path=''):
    files = glob.glob(folder)
    seg = Seg()
    for jpg_name in tqdm(files):
        try:
            seged_name = os.path.basename(jpg_name).replace('.jpg', '.png')
            seged_name = os.path.join(seg_path, seged_name)

            gsv_img = Image.open(jpg_name)

            seged = seg.getSeg(gsv_img, seged_name)

            colored_name = os.path.basename(jpg_name).replace('.jpg', '_color.png')
            colored_name = os.path.join(color_path, colored_name)
            seg.getColor(seged, dataset='ade20k', saved_name=colored_name)
        except Exception as e:
            print("Error in seg_files():", jpg_name, e)

        #colored.save(colored_name)

if __name__ == '__main__':

    folder =     r'K:\Research\Trees\NewYorkCity_test\google_street_images\*.jpg'
    seg_path =   r"K:\Research\Trees\NewYorkCity_test\tree_seg"
    color_path = r"K:\Research\Trees\NewYorkCity_test\tree_color"

    seg_files(folder, seg_path=seg_path, color_path=color_path)

    # c
    # filename = r'X:\Shared drives\sidewalk\Street_trees\Philly\tree_jpg\_-_elg9jYBI22BgLxPW4Xw_-75.148922_39.965694_0_173.jpg'
    # img = image.imread(filename)
    # predict = seg.getSeg(img)
    # print(predict)
    # colored = seg.getColor(predict)
    # print(colored)
    # plt.imshow(colored)
    # plt.show()
