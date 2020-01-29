"""
Designed by Huan Ning, gladcolor@gmail.com, 2020.01.14

"""
import glob
from pyproj import Proj, transform, itransform
from shapely import affinity
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
import sys
import fiona
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
from shapely.geometry import Point, Polygon, mapping, LineString, MultiLineString
import shapely
import csv
from skimage import io
from PIL import features
import urllib.request
import urllib
from geopy.distance import geodesic

from shapely.geometry import Polygon
# from centerline.geometry import Centerline
import logging
from label_centerlines import get_centerline
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'
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
        ID, left, bottom, right, top = bound
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
    """

    :param jdata:
    :param bearing:
    :return: panoId, yawDeg
    """
    try:
        crt_panoId = jdata['Location']['panoId']
        links = jdata["Links"]
    except Exception as e:
        print("Error in getting Links in json:", e)
        return 0, -999
    bearing = float(bearing)
    yaw_in_links = [float(link['yawDeg']) for link in links]
    diff = [abs(yawDeg - bearing) for yawDeg in yaw_in_links]
    idx = diff.index(min(diff))
    panoId = links[idx]['panoId']
    yawDeg = float(links[idx]['yawDeg'])

    if (panoId == crt_panoId) and (len(links) > 1):
        diff.pop(idx)
        links.pop(idx)
        idx = diff.index(min(diff))
        panoId = links[idx]['panoId']
        yawDeg = float(links[idx]['yawDeg'])
        return panoId, yawDeg

    if len(links) == 1:
        if abs(float(links[0]['yawDeg']) - float(bearing)) < 45:
            # print("abs(links[0]['yawDeg'] - bearing)")
            return links[0]['panoId']
            # print("abs(links[0]['yawDeg'] - bearing)")
        else:
            return 0, -999
    else:
        return panoId, yawDeg

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
    suffix_worldfile = os.path.basename(img_file)[-3] + os.path.basename(img_file)[-1] + 'w'
    suffix_worldfile = suffix_worldfile.lower()


    world_file_name = img_file[:-3] + suffix_worldfile
    world_coords = np.array([1, 0, 0, 0, 1, 0])
    if os.path.exists(world_file_name):
        f = open(world_file_name, 'r')
        coords = f.readlines()
        coords = [float(i) for i in coords]
        world_coords = np.array(coords)
        f.close()
    # world_coords = world_coords.reshape((3, 2))
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

    # BUG of get_centerline(): it cannot process large coordinates more than 6 digits
    contours = [np.squeeze(cont) for cont in contours]
    # print(contours[0])

    # contours = [np.c_[cont, np.ones((len(cont),))] for cont in contours]
    # print(contours[0])
    # contours = [np.dot(cont, world_coords) for cont in contours]
    # print(contours[0])
    # print(contours)


    return contours, world_coords
    # cnt = contours[3]
    # print(contours)
    # backtorgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
    # cv2.imshow("Original", backtorgb)
    # t = cv2.drawContours((backtorgb), contours, -1, (0, 255, 0), 3)
    # print(len(contours))
    #
    # cv2.imshow('closed', t)
    # if cv2.waitKey(0) == ord('q'):
    #     cv2.destroyAllWindows()

def get_polygon_centline(contours, world_coords=[], segmentize_maxlen=8, max_points=3000, simplification=0.05, smooth_sigma=5):
    if not isinstance(contours, list):
        contours = [contours]
    results = []    
    for contour in contours:
        try:    
            polygon = Polygon(contour)
            # print(polygon)
            centerlines = get_centerline(polygon, segmentize_maxlen=segmentize_maxlen, max_points=max_points, simplification=simplification, smooth_sigma=smooth_sigma)
            # print(centerline1)
            results.append(centerlines)
        except Exception as e:
            print("Error in get_polygon_centline():", e)
            results.append(0)
            continue

    if len(world_coords) > 0:
        # coords =
        # centerlines = [np.c_[cont, np.ones((len(cont),))] for cont in contours]
        results = [shapely.affinity.affine_transform(cont, world_coords) for cont in results]
        print(results)

    return results

def save_geometry_to_shp(saved_name, geos, schema, attributes):
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    if len(geos) > 0:
        sink = fiona.open(saved_name, 'w', 'ESRI Shapefile', schema)
    for idx, geo in enumerate(geos):
        # **source.meta is a shortcut to get the crs, driver, and schema
        # keyword arguments from the source Collection.
        try:

            sink.write({'geometry': mapping(geo), 'properties': attributes[idx]})
        except Exception as e:
            # Writing uncleanable features to a different shapefile
            # is another option.
            logging.exception("Error cleaning feature %s:", attributes[idx])

def getNextShot(crt_json, crt_yaw,  direction_d, phi=90, saved_path='', pitch=0, width=1024, height=768):

    """

    :param crt_json:
    :param shot_angle:
    :param direction_d: the direction needs to go
    :param phi: from the heading angle of panoramo
    :param saved_path:
    :return:
    """

    next_panoId, yawDeg = getSameDirectionPanoId(crt_json, direction_d)
    yawDeg = float(yawDeg)
    if yawDeg == -999:
        return 0, 0

    next_json = gpano.getJsonfrmPanoID(next_panoId)
    lon = float(next_json['Location']['lng'])
    lat = float(next_json['Location']['lat'])
    # yawDeg = next_json['Location']['lat']
    pano_yaw_deg = float(next_json['Projection']['pano_yaw_deg'])

    is_right_side = True

    angle_crt_yaw_direction_d = crt_yaw - direction_d
    if angle_crt_yaw_direction_d < 0:
        angle_crt_yaw_direction_d += 360

    if angle_crt_yaw_direction_d > 180:
        is_right_side = False
        phi = phi + 180

    shotYaw = yawDeg + phi
    if shotYaw > 360:
        shotYaw -= 360

    # if abs(yawDeg + phi - crt_yaw) > abs(yawDeg + phi + 180 - crt_yaw):
    #     shotYaw = yawDeg + phi + 180 - crt_yaw
    # else:
    #     shotYaw = yawDeg + phi + 180 - crt_yaw

    print("Sidewalk following shoots angles(deg): phi", phi)
    print("Sidewalk following shoots angles(deg): crt_yaw", crt_yaw)
    print("Sidewalk following shoots angles(deg): yawDeg", yawDeg)
    print("Sidewalk following shoots angles(deg): shotYaw", shotYaw)


    next_img, file_name = gpano.getImagefrmAngle(lon, lat, saved_path=saved_path, prefix=next_panoId, suffix='', width=1024, height=768,
                         pitch=0, yaw=shotYaw)
    return next_img, file_name, yawDeg

def go_along_sidewalk(lon, lat, bearing_deg, pixel_thres, rtree_bounds, rtree_dangles, saved_path='sidewalk_imgs', distance_thres=20):
    """

    :param lon:
    :param lat:
    :param bearing_deg:
    :param pixel_thres: 30 for 0.1m resolution
    :param rtree_bounds: boundaries of Area of Interest
    :param saved_path:
    :param distance_thres: meter
    :return:
    """
    try:
        steps_lonlat = []

        lon_d = lon
        lat_d = lat

        if not os.path.exists(saved_path):
            os.mkdir(saved_path)

        sidewalk_num = 99999
        dangleImg, jpg_name = gpano.shootLonlat(lon, lat, saved_path=saved_path)

        try:
            if dangleImg == 0:
                print("Did not found json in lon/lat : ", lon, lat)
                return steps_lonlat
        except:
            return steps_lonlat

        if shotExists(jpg_name, folder=saved_path, threshold=15):
            print("Found an existing shoot: ", jpg_name)
            return steps_lonlat

        basename = os.path.basename(jpg_name)
        params = basename[:-4].split('_')
        lon = float(params[-4])
        lat = float(params[-3])
        # print("params:", params)
        if geodesic((lat, lon), (lat_d, lon_d)).m > distance_thres:  # (latitude, longitude) don't confuse
            print("Distance is too long : meter, (lat, lon), (lat_d, lon_d): ", geodesic((lat, lon), (lat_d, lon_d)).m, (lat, lon), (lat_d, lon_d))
            return steps_lonlat

        panoId = '_'.join(params[:(len(params) - 4)])

        crt_json = gpano.getJsonfrmPanoID(panoId)
        crt_yaw = float(params[-1])  # shoting direction
        print("Sidewalk first shoot to bearing (deg):", crt_yaw)

        seged_name = jpg_name.replace('.jpg', '.png')
        seged = seg.getSeg(dangleImg, seged_name)

        landcover, landcover_file = gsv.seg_to_landcover2(seged_name, saved_path)

        colored_name = seged_name.replace('.png', '_seg_color.png')
        # colored = seg.getColor(seged, dataset='ade20k', saved_name=colored_name)

        sidewalk_idx = []
        for label in LABEL_IDS:
            sidewalk_idx.append(np.argwhere((landcover == label)))

        sidewalk_idx = np.concatenate(sidewalk_idx)

        if len(sidewalk_idx) < pixel_thres:  # 3 m2,
            # print('No sidewalk.')
            # dangles_results_mp.append(("Found sidewalk", panoId, len(sidewalk_idx)))
            print('Found no sidewalk at: ', lon, lat, len(sidewalk_idx), jpg_name)
            return steps_lonlat

        steps_lonlat.append([panoId, lon, lat])

            # dangles_results_mp.append(("Found sidewalk", panoId))
            # landcover = gsv.seg_to_landcover2(seged_name, saved_path)
            # landcover = np.array(landcover)  # for unkown reason to get a tuple (np.array)
            # colored_name = seged_name.replace('.png', '_seg_color.png')
            # colored = seg.getColor(seged, dataset='ade20k', saved_name=colored_name)
            # colored.i
            # plt.imshow(colored)
            # plt.show()

        # else:
        #     # dangles_results_mp.append(("No sidewalk", panoId, 0))
        #     print('No sidewalk.')
        #     return 0


        x, y = gsv.lonlat_to_proj(lon, lat, 6565, 4326)
        isInside = isInBounds((x, y, x, y), rtree_bounds)
        while isInside: # if inside the boudaries
            next_img, jpg_name, yawDeg = getNextShot(crt_json, crt_yaw, bearing_deg, phi=90, saved_path=saved_path, pitch=0, width=1024, height=768)
            # pano_box = getPanoBox(lon, lat)
            try:
                if next_img == 0:
                    print("Did not found json in lon/lat : ", lon, lat)
                    break
            except:
                pass

            if shotExists(jpg_name, folder=saved_path, threshold=15):
                print("Found an existing shoot: ", jpg_name)
                return steps_lonlat

            basename = os.path.basename(jpg_name)
            params = basename[:-4].split('_')
            # bearing_deg = float(params[-1])
            panoId = '_'.join(params[:(len(params) - 4)])

            lat_d = lat
            lon_d = lon

            lon = float(params[-4])
            lat = float(params[-3])

            if geodesic((lat, lon), (lat_d, lon_d)).m > distance_thres:  # (latitude, longitude) don't confuse
                print("Distance is too long : meter, (lat, lon), (lat_d, lon_d): ",
                      geodesic((lat, lon), (lat_d, lon_d)).m, (lat, lon), (lat_d, lon_d))

                return steps_lonlat

            seged_name = jpg_name.replace('.jpg', '.png')
            seged = seg.getSeg(next_img, seged_name)

            landcover, landcover_file = gsv.seg_to_landcover2(seged_name, saved_path)
            colored_name = seged_name.replace('.png', '_seg_color.png')
            # colored = seg.getColor(seged, dataset='ade20k', saved_name=colored_name)

            sidewalk_idx = []
            for label in LABEL_IDS:
                sidewalk_idx.append(np.argwhere((landcover == label)))

            sidewalk_idx = np.concatenate(sidewalk_idx)

            if len(sidewalk_idx) > pixel_thres:  # 3 m2,
                # dangles_results_mp.append(("Found sidewalk", panoId, len(sidewalk_idx)))

                steps_lonlat.append([panoId, lon, lat])

                print('Found sidewalk at: ', lon, lat, len(sidewalk_idx), jpg_name)
                view_box = getPanoBox(lon, lat, bearing_deg)



                # x, y = view_box.exterior.xy
                # fig = plt.figure(1, figsize=(5, 5), dpi=90)
                # ax = fig.add_subplot(111)
                # ax.plot(x, y)

                isConnected = isInBounds(view_box.bounds, rtree_dangles)

                if isConnected:
                    print("Connected to another dangles.")
                    return steps_lonlat
                else:
                    print('Found few sidewalks at: ', lon, lat, len(sidewalk_idx), jpg_name)

                # dangles_results_mp.append(("Found sidewalk", panoId))
                landcover = gsv.seg_to_landcover2(seged_name, saved_path)
                colored_name = seged_name.replace('.png', '_seg_color.png')
                colored = seg.getColor(seged, dataset='ade20k', saved_name=colored_name)

                lon = params[-4]
                lat = params[-3]



                crt_yaw = float(params[-1])
                # crt_yaw =
                bearing_deg = yawDeg
                x, y = gsv.lonlat_to_proj(lon, lat, 6565, 4326)
                isInside = isInBounds((x, y, x, y), rtree_bounds)
                crt_json = gpano.getJsonfrmPanoID(panoId)
                # jdata = gpano.getJsonfrmPanoID(panoId)

                # go_along_sidewalk(lon, lat, bearing_deg, pixel_thres, rtree_bounds, rtree_dangles,\
                #                   saved_path=saved_path)
                # print('steps_lonlat: ', steps_lonlat)
                # steps_lonlat_np = np.array(steps_lonlat) * 100
                # plt.scatter(steps_lonlat_np[:, 0], steps_lonlat_np[:, 1])
                # plt.show()
            else:  #len(sidewalk_idx) < pixel_thres:
                lon = params[-4]
                lat = params[-3]
                print('Found no sidewalks at: ', lon, lat, len(sidewalk_idx), jpg_name)
                return steps_lonlat

        return steps_lonlat

    except Exception as e:
        print("Error in go_along_sidewalk():", e)
        return steps_lonlat

def getPanoBox(lon, lat, bearing_deg, w_meter=8, h_meter=15):
    """
    :param lon:
    :param lat:
    :bearing_deg:
    :return: a shapely Polygon

    """
    proj_local = Proj(f'+proj=tmerc +lon_0={lon} +lat_0={lat}')
    #     x, y = transform(Proj('epsg:4326'), proj_local, lat, lon) #(0, 0)

    upper_left = (0 - w_meter / 2, h_meter)
    upper_right = (0 + w_meter / 2, h_meter)
    bottom_right = (0 + w_meter / 2, 0)
    bottom_left = (0 - w_meter / 2, 0)

    box = Polygon((upper_left, upper_right, bottom_right, bottom_left))
    box = affinity.rotate(box,
                          -bearing_deg)  # Positive angles are counter-clockwise and negative are clockwise rotations.

    box_pts = box.exterior.coords
    box_pts = list(box_pts)
    results = itransform(proj_local, Proj('epsg:4326'), box_pts)
    results = list(results)
    results = [(c[1], c[0]) for c in results]

    return Polygon(results)

def shotExists(jpg_name, folder, threshold=15):

    basename = os.path.basename(jpg_name)
    params = basename[:-4].split('_')
    panoId = '_'.join(params[:(len(params) - 4)])
    shotDirection = float(params[-1])

    candidates = glob.glob(os.path.join(folder, panoId + '*.jpg'))
    for can in candidates:
        basename_c = os.path.basename(can)
        params_c = basename_c[:-4].split('_')
        panoId_c = '_'.join(params_c[:(len(params) - 4)])
        shotDirection_c = float(params_c[-1])

        if abs(shotDirection_c - shotDirection) < threshold:
            return True

    return False

def go_along_sidewalk_excute():
    print("go_along_sidewalk_excute()...")
    # dangle_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\DVRPC\Dangles.csv'
    # getBearingAngle(dangle_file)
    saved_path = 'I:\DVRPC\Fill_gap\StreetView\images5'
    bearing_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\DVRPC\Dangles_bearing.csv'
    dangles = pd.read_csv(bearing_file).iloc[459:] # index =  row_in_the_dangle_table - 1. E.g., iloc[11475] = id 11476 in the table
    # print(dangles)    # 38:40

    dangles_latlon_mp = mp.Manager().list()
    dangles_results_mp = mp.Manager().list()
    for idx, row in dangles.iterrows():
        dangles_latlon_mp.append((idx, row['POINT_X'], row['POINT_Y'], row['bearing']))

    total_num = len(dangles_latlon_mp)
    while len(dangles_latlon_mp) > 0:
        idx, lon_d, lat_d, bearing_deg = dangles_latlon_mp.pop(0)
        pixel_thres = 30
        print("Processing row: ", idx)

        rtree_dangles = load_RtreeIdx(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\DVRPC\Dangles_rtree')
        rtree_bounds = load_RtreeIdx(r'I:\DVRPC\Test49rtree')
        sidewalk_panos = go_along_sidewalk(lon_d, lat_d, bearing_deg, pixel_thres, rtree_bounds, rtree_dangles, saved_path=saved_path)


        print('lon, lat, bearing: ', lon_d, lat_d, bearing_deg)

        with open(os.path.join(saved_path, str(idx) + r'.txt'), 'w') as f:
            for p in sidewalk_panos:
                f.writelines(str(p) + '\n')



        # jdata = gpano.getPanoJsonfrmL

        #     print("Did not found json in lon/lat : ", lon_d, lat_d)

    print("go_along_sidewalk_excute() done.")


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
            if len(sidewalk_idx) > 30:  # 3 m2,
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

if __name__ == "__main__":
    print("Starting to fill sidewalks...")

    # main()

    go_along_sidewalk_excute()

    # Testing shape file generation
    '''    
    img_file = r'I:\DVRPC\Fill_gap\StreetView\images\sG51XsNz_X9EWGv_nU8jaw_-74.848704_40.150078_0_134.31_landcover.png'
    contours, world_coords = getContours(img_file)
    # print(contours)
    centerlines = get_polygon_centline(contours, world_coords)
    # print(centerlines)
    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
    }

    attrs = [{'id': i} for i in range(len(centerlines))]

    save_geometry_to_shp(r'I:\test.shp', centerlines, schema, attrs)
    # print(list(centerlines[0].coords))
    '''