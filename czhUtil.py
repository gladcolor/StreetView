import math
import struct

import base64
import os
import zlib
from pyproj import Proj, transform
from PIL import Image

# from open3d import *
from geographiclib.geodesic import Geodesic

def lonlat2WebMercator( lon, lat):
    return transform(Proj(init='epsg:4326'), Proj(init='epsg:2824'), lon, lat)


def WebMercator2lonlat(X, Y):
    return transform(Proj(init='epsg:2824'), Proj(init='epsg:4326'), X, Y)

def get_bin(a):
    ba = bin(a)[2:]
    return "0" * (8 - len(ba)) + ba

def getUInt16(arr, ind):
    a = arr[ind]
    b = arr[ind + 1]
    return int(get_bin(b) + get_bin(a), 2)

def getFloat32(arr, ind):
    return bin_to_float("".join(get_bin(i) for i in arr[ind: ind + 4][::-1]))

def bin_to_float( binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

def parse(b64_string):
    # fix the 'inccorrect padding' error. The length of the string needs to be divisible by 4.
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    # convert the URL safe format to regular format.
    data = b64_string.replace("-", "+").replace("_", "/")

    data = base64.b64decode(data)  # decode the string
    data = zlib.decompress(data)  # decompress the data

    return data

def get_color_pallete( npimg, dataset='ade20k'):
    # Huan changed the label 1 from 120, 120, 120 to 0, 0, 0
    adepallete = [
        0, 0, 0, 0, 0, 0, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204,
        5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6,
        82,
        143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255,
        255,
        7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184,
        6,
        10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0,
        255,
        20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10,
        15,
        20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173,
        255,
        31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255,
        163,
        0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173,
        255,
        0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184,
        0,
        31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255,
        0,
        194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255,
        0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255,
        255,
        0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0,
        163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,
        10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41,
        0,
        255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0,
        133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255]

    cityspallete = [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        102, 102, 156,
        190, 153, 153,
        153, 153, 153,
        250, 170, 30,
        220, 220, 0,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        255, 0, 0,
        0, 0, 142,
        0, 0, 70,
        0, 60, 100,
        0, 80, 100,
        0, 0, 230,
        119, 11, 32,
    ]
    """Visualize image.
            Parameters
            ----------
            npimg : numpy.ndarray
                Single channel image with shape `H, W, 1`.
            dataset : str, default: 'pascal_voc'
                The dataset that model pretrained on. ('pascal_voc', 'ade20k')
            Returns
            -------
            out_img : PIL.Image
                Image with color pallete
    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255
    # put colormap
    if dataset == 'ade20k':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(adepallete)
        return out_img
    elif dataset == 'citys':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityspallete)
        return out_img
    out_img = Image.fromarray(npimg.astype('uint8'))
    vocpallete = getvocpallete(256)
    out_img.putpalette(vocpallete)
    return out_img

def getvocpallete( num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete

# get all the fillpath in the directory  include sub-directory
def getfilenamefromfilepath(curDir, filename_list, ext=('tif', 'tiff','png')):

    if os.path.isfile(curDir):
        if curDir.lower().endswith(ext):
            file_path,file_ext = os.path.splitext(curDir)
            filename = os.path.basename(file_path)
            filename_list.append(filename)
    else:
        dir_or_files = os.listdir(curDir)
        for dir_file in dir_or_files:
            dir_file_path = os.path.join(curDir, dir_file)

            # check is file or directory
            if os.path.isdir(dir_file_path):
                getfilenamefromfilepath(dir_file_path, filename_list, ext)
            else:
                # extension_ = dir_file_path.split('.')[-1]
                # if (extension_.lower() in ext):

                if dir_file_path.lower().endswith(ext):
                    file_path, file_ext = os.path.splitext(dir_file_path)
                    filename = os.path.basename(file_path)
                    filename_list.append(filename)

