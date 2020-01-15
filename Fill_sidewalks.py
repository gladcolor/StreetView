"""
Designed by Huan Ning, gladcolor@gmail.com, 2020.01.14

"""
from pyproj import Proj, transform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
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

# WINDOWS_SIZE = '100, 100'
# chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
# Loading_time = 5
import sqlite3
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

def getRtreeIdx(retangles, saved_path):
    rtreeIdx = 0
    return rtreeIdx

def isNearPts(pt, pts):
    idx = 0
    return idx


def main():
    print("Starting main()...")
    # dangle_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\DVRPC\Dangles.csv'
    # getBearingAngle(dangle_file)
    bearing_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\DVRPC\Dangles_bearing.csv'
    dangles = pd.read_csv(bearing_file).iloc[33:43]
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
            
            plt.imshow(dangleImg)
            plt.show()

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
                plt.imshow(colored)
                plt.show()

            else:
                dangles_results_mp.append(("No sidewalk", panoId, 0))
                print('No sidewalk.')

        except Exception as e:
            print("Error in processing json: ", e)
            continue

    print("dangles_results_mp: ", dangles_results_mp)
    #

    print("Finished main().")

if __name__ == "__main__":
    print("Starting to fill sidewalks...")
    main()

