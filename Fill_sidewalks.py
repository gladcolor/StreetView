"""
Designed by Huan Ning, gladcolor@gmail.com, 2020.01.14

"""
from pyproj import Proj, transform
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib as plt
from scipy import interpolate
import matplotlib.cm as cm
import multiprocessing as mp
import numpy as np
import scipy.ndimage
from scipy import interpolate
from math import *
import pandas as pd
# import selenium
import os
import time
from io import BytesIO
import pandas as pd
import random
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
from PIL import Image
# import requests
import json
import csv
import math
import sys
import base64
import zlib
import cv2
import struct
import matplotlib.pyplot as plt
import PIL
from shapely.geometry import Point, Polygon
import csv
from skimage import io
from PIL import features
import urllib.request
import urllib
from shapely.geometry import Polygon
# from centerline.geometry import Centerline

from label_centerlines import get_centerline

# WINDOWS_SIZE = '100, 100'
# chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
# Loading_time = 5
# import sqlite3
import GPano
import Segmentor_gluon
from rtree import index

gpano = GPano.GPano()
gsv = GPano.GSV_depthmap()
seg = Segmentor_gluon.Seg()

LABEL_IDS = [11, 52]
saved_path = r'I:\DVRPC\Fill_gap\StreetView\images'

def getBearingAngle(dangle_file):
    pts = pd.read_csv(dangle_file)
    # print(pts)
    for idx, row in pts.iterrows():
        if idx % 2 == 0:
            row_next = pts.iloc[idx + 1]
            radLatA = math.radians(row.POINT_Y)
            radLonA = math.radians(row.POINT_X)
            radLatB = math.radians(row_next.POINT_Y)
            radLonB = math.radians(row_next.POINT_X)
            angle = gpano.getDegreeOfTwoLonlat(radLatA, radLonA, radLatB, radLonB)
            pts.loc[idx, 'bearing'] = angle
            pts.loc[idx + 1, 'bearing'] = angle
            # print(idx, pts.loc[idx, 'bearing'])
    pts2 = pts[1::2]
    pts2.to_csv(dangle_file.replace('.csv', '_bearing.csv'), index=False)

    print("getBearingAngle() saved bearning file in: ", dangle_file.replace('.csv', '_bearing.csv'))

def build_RtreeIdx(bounds, saved_name):  # saved_name has no suffix
    """

    :param bounds: a list of (id, left, bottom, right, top)
    :param saved_name: saved_name has no suffix
    :return: R-tree
    """
    r_idx = index.Index(saved_name)
    for bound in bounds:
        ID, left, bottom, right, top = bounds
        # print( ID, left, bottom, right, top)
        r_idx.insert(ID, (left, bottom, right, top))
    r_idx.close()
    return r_idx

def load_RtreeIdx(saved_name):  # saved_name has no suffix
    """
    :param saved_name: saved_name has no suffix
    :return: R-tree
    """
    r_idx = index.Index(saved_name)
    return r_idx

def isNearPts(pt, pts):
    idx = 0
    return idx

def getSameDirectionPanoId(jdata, bearing):
    try:
        crt_panoId = float(jdata['Projection']['pano_yaw_deg'])
        links = jdata["Links"]
    except Exception as e:
        print("Error in getting Links in json:", e)
        return 0

    yaw_in_links = [float(link['yawDeg']) for link in links]
    diff = [abs(yawDeg - bearing) for yawDeg in yaw_in_links]
    idx = diff.index(min(diff))
    panoId = links[idx]['panoId']

    if (panoId == crt_panoId) and (len(links) > 1):
        diff.pop(idx)
        links.pop(idx)
        idx = diff.index(min(diff))
        panoId = links[idx]['panoId']
        return panoId

    if len(links) == 1:
        if abs(float(links[0]['yawDeg']) - float(bearing)) < 45:
            # print("abs(links[0]['yawDeg'] - bearing)")
            return links[0]['panoId']
            # print("abs(links[0]['yawDeg'] - bearing)")
        else:
            return 0
    else:
        return panoId

def isInBounds(bounds, Rtree_idx):
    """

    :param bounds: (left, bottom, right, top)
    :param Rtree_idx:
    :return:
    """
    intersects = len(list(Rtree_idx.intersection(bounds)))
    if intersects > 0:
        return True
    else:
        return False

def getContours(img_file, kernel=9):
    """

    :param file_path:
    :return: Shapely Polygons
    """
    # img_cv = cv2.imread(img_file)
    # img_io = io.imread(img_file)
    img_pil = Image.open(img_file)
    # plt.imshow(img_io)
    # plt.show()
    np_img = np.array(img_pil)
    #
    # unique_elements, counts_elements = np.unique(np_img, return_counts=True)
    # print(unique_elements)
    # print(counts_elements)

    np_img = np.where((np_img == 12) | (np_img == 53), 255, 0).astype(np.uint8)

    # unique_elements, counts_elements = np.unique(np_img, return_counts=True)
    # print(unique_elements)
    # print(counts_elements)

    pimg = Image.fromarray(np_img)

    # plt.imshow(pimg)
    # plt.show()

    # cannot convert png to cv2 format.
    pimg.save("temp.png")

    # img1 = Image.open("temp.png")
    # plt.imshow(img1)
    # plt.show()

    img_cv2 = cv2.imread("temp.png", cv2.IMREAD_UNCHANGED)

    # cv2.imshow('landcover', cv2.imread(img_file, cv2.IMREAD_UNCHANGED))

    # plt.imshow(img_cv2)
    # plt.show()
    #
    # img_np = np.array(img_cv2)
    # unique_elements, counts_elements = np.unique(img_np, return_counts=True)
    # print(unique_elements)
    # print(counts_elements)

    # cv2.imshow('MORPH_CLOSE', cv2.imread("temp.png", cv2.IMREAD_UNCHANGED))

    ret, thresh = cv2.threshold(img_cv2, 0, 255, 0)
    # cv2.imshow('thresh', thresh)

    g = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, g)
    # cv2.imshow('MORPH_CLOSE', closed)

    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, g)
    # cv2.imshow('MORPH_OPEN', opened)

    # backtorgb = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2RGB)
    # print(thresh[:, 100

    # plt.imshow(img_cv2)
    # plt.show()

    contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [np.squeeze(cont) for cont in contours]

    # print(contours)
    return contours
    # cnt = contours[3]
    # print(contours)
    backtorgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
    # cv2.imshow("Original", backtorgb)
    # t = cv2.drawContours((backtorgb), contours, -1, (0, 255, 0), 3)
    # print(len(contours))
    #
    # cv2.imshow('closed', t)
    # if cv2.waitKey(0) == ord('q'):
    #     cv2.destroyAllWindows()

def get_polygon_centline(contours, segmentize_maxlen=8, max_points=3000, simplification=0.05, smooth_sigma=5):
    if not isinstance(contours, list):
        contours = [contours]
    results = []    
    for contour in contours:
        try:    
            polygon = Polygon(contour)
            # print(polygon)
            centerline1 = get_centerline(polygon, segmentize_maxlen=segmentize_maxlen, max_points=max_points, simplification=simplification, smooth_sigma=smooth_sigma)
            # print(centerline1)
            results.append(centerline1)
        except Exception as e:
            print("Error in get_polygon_centline():", e)
            results.append(0)
            continue
    return results


def main():
    print("Starting main()...")
    # dangle_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\DVRPC\Dangles.csv'
    # getBearingAngle(dangle_file)
    bearing_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\DVRPC\Dangles_bearing.csv'
    dangles = pd.read_csv(bearing_file).iloc[:]
    # print(dangles)


    dangles_latlon_mp = mp.Manager().list()
    dangles_results_mp = mp.Manager().list()
    for idx, row in dangles.iterrows():
        dangles_latlon_mp.append((idx, row['POINT_X'], row['POINT_Y'], row['bearing']))

    total_num = len(dangles_latlon_mp)
    while len(dangles_latlon_mp) > 0:
        idx, lon_d, lat_d, bearing = dangles_latlon_mp.pop(0)

        # print('lon, lat, bearing: ', lon, lat, bearing)
        print("Processing row: ", idx)
        # jdata = gpano.getPanoJsonfrmLonat(lon_d, lat_d)
        # if jdata == 0:
        #     print("Did not found json in lon/lat : ", lon_d, lat_d)
        #     continue

        try:
            # lon_g = jdata['Location']['lng']
            # lat_g = jdata['Location']['lat']

            dangleImg, jpg_name = gpano.shootLonlat(lon_d, lat_d, saved_path=saved_path)
            # print(type(dangleImg))
            try:
                if dangleImg == 0:
                    print("Did not found json in lon/lat : ", lon_d, lat_d)
                    dangles_results_mp.append(("No street view image", 'No pano', -1))
                    continue
            except:
                pass
            
            # plt.imshow(dangleImg)
            # plt.show()

            basename = os.path.basename(jpg_name)
            params = basename[:-4].split('_')
            # print("params:", params)
            panoId = '_'.join(params[:(len(params) - 4)])

            seged_name = jpg_name.replace('.jpg', '.png')
            seged = seg.getSeg(dangleImg, seged_name)
            sidewalk_idx = []
            for label in LABEL_IDS:
                sidewalk_idx.append(np.argwhere((seged == label)))
            sidewalk_idx = np.concatenate(sidewalk_idx)
            if len(sidewalk_idx) > 200:
                dangles_results_mp.append(("Found sidewalk", panoId, len(sidewalk_idx)))

                print('Found sidewalk: ', panoId, idx, len(sidewalk_idx))

                dangles_results_mp.append(("Found sidewalk", panoId))

                landcover = gsv.seg_to_landcover2(seged_name, saved_path)
                colored_name = seged_name.replace('.png', '_seg_color.png')
                colored = seg.getColor(seged, dataset='ade20k', saved_name=colored_name)
                # colored.i
                # plt.imshow(colored)
                # plt.show()

            else:
                dangles_results_mp.append(("No sidewalk", panoId, 0))
                print('No sidewalk.')

        except Exception as e:
            print("Error in processing json: ", e)
            continue

    print("dangles_results_mp: ", dangles_results_mp)
    f = open(os.path.join(saved_path, 'results.txt'), 'w')
    for row in dangles_results_mp:
        f.writelines(row + '\n')
    f.close()
    #

    print("Finished main().")

if __name__
    print("Starting to fill sidewalks...")
    # main()

    img_file = r'I:\DVRPC\Fill_gap\StreetView\images\sG51XsNz_X9EWGv_nU8jaw_-74.848704_40.150078_0_134.31_landcover.png'
    contours = getContours(img_file)
    centerlines = get_polygon_centline(contours)
    print(list(centerlines[0].coords))
