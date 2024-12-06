"""
Designed by Huan Ning, gladcolor@gmail.com, 2020.12.28
"""

# Python built-ins
import os
import time
import json
import csv
import math
from math import *
import sys
import struct
import base64
import zlib
import multiprocessing as mp
from skimage import morphology
from pyproj import CRS
from pyproj import Transformer

from sklearn.linear_model import LinearRegression
from PIL import ImageFilter
# Geospatial processing
from pyproj import Proj, transform
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from scipy.ndimage import gaussian_filter

import numpy as np
from numpy import inf

# import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from skimage import io
import yaml

import PIL
from PIL import Image, features
from PIL import Image, ImageDraw

import cv2

from io import BytesIO
import random

import requests
import urllib.request
import urllib
import logging


import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt


# import selenium
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
import sqlite3
from bs4 import BeautifulSoup
# WINDOWS_SIZE = '100, 100'
# chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
# Loading_time = 5

# import utils0
import utils

import logging.config


logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='Pano.log',
                filemode='w')


GROUND_VECTOR_THRES = 10
DEM_SMOOTH_SIGMA    = 1
DEM_COARSE_RESOLUTION = 0.8

def setup_logging(default_path='log_config.yaml', logName='', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if logName !='':
                config["handlers"]["file"]['filename'] = logName
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

yaml_path = 'log_config.yaml'
setup_logging(yaml_path)
logger = logging.getLogger('LOG.file')


logging.shutdown()

class GSV_pano(object):
    def __init__(self, panoId=0, json_file='', request_lon=None, request_lat=None,
                 request_address='', crs_local=None, saved_path=''):
        self.panoId = panoId  # test case: BM1Qt23drK3-yMWxYfOfVg
        self.request_lon = request_lon  # test case: -74.18154077638651
        self.request_lat = request_lat  # test case: 40.73031168738437
        self.request_address = request_address # test case: '93-121 W Kinney St, Newark, NJ 07102'  # not finished yet.
        self.jdata = None  # json object
        self.lat = None
        self.lon = None
        self.saved_path = saved_path
        self.crs_local = crs_local
        self.json_file = json_file
        self.x = None
        self.y = None
        self.z = None
        self.normal_height = None


        # image storage: numpy array
        self.depthmap = {"depthMap":None,
                         'dm_mask': None,
                         "width":None,
                         "height": None,
                         "plane_idx_map": None,
                         "normal_vector_map": None,
                         "ground_mask": None,
                         "zoom": None}
        self.panorama ={"image": None, "zoom": None}
        self.segmenation ={"segmentation": None, "zoom": None, 'full_path':None}
        self.point_cloud = {"point_cloud": None, "zoom": None, "dm_mask": None}
        self.DEM = {"DEM": None,
                    "zoom": None,
                    "resolution": None,
                    "elevation_wgs84_m": None,
                    "elevation_egm96_m": None,
                    "camera_height": None
                    }
        self.DOM = {"DOM": None, "zoom": None, "resolution": None,
                    "DOM_points": None} #

        try:

            if (self.panoId != 0) and (self.panoId is not None) and (len(str(self.panoId)) == 22):
                # print("panoid: ", self.panoId)
                self.jdata = self.getJsonfrmPanoID(panoId=self.panoId, dm=1, saved_path=self.saved_path)
                self.lon = self.jdata['Location']['lng']
                self.lat = self.jdata['Location']['lat']
            # else:
            #     logging.info("Found no paoraom in GSV_pano _init__(): %s" % panoId)

            if request_lat and request_lon:
                if (-180 <= request_lon <= 180) and (-90 <= request_lat <= 90):
                    self.panoId, self.lon, self.lat = self.getPanoIDfrmLonlat(request_lon, request_lat)



            if os.path.exists(self.json_file):
                try:
                    with open(self.json_file, 'r') as f:
                        jdata = json.load(f)
                        self.jdata = jdata
                        self.panoId = self.jdata['Location']['panoId']
                        self.lon = self.jdata['Location']['lng']
                        self.lat = self.jdata['Location']['lat']

                except Exception as e:
                    logging.info("Error in GSV_pano _init__() when loading local json file: %s, %s", self.json_file, e)
            # else:
            #     basename = os.path.basename(json_file)[:22]
            #     if panoId == 0:
            #         panoId = basename
            #         self.panoId = panoId







            # if self.crs_local and (self.lat is not None) and (self.lon is not None ):
            #     transformer = utils.epsg_transform(in_epsg=4326, out_epsg=self.crs_local)
            #     self.x, self.y = transformer.transform(self.lat, self.lon)



        except Exception as e:
            logging.exception("Error in GSV_pano _init__(): %s", e)


    def getPanoIDfrmLonlat(self, lon, lat):
        url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
        url = url.format(lat, lon)
        resp = requests.get(url, proxies=None)
        # print(url)
        line = resp.text.replace("/**/_xdc_._v2mub5 && _xdc_._v2mub5( ", "")[:-2]

        if len(line) > 1000:
            try:
                jdata = json.loads(line)
                panoId = jdata[1][1][1]
                pano_lon = jdata[1][5][0][1][0][3]
                pano_lat = jdata[1][5][0][1][0][2]
                return panoId, pano_lon, pano_lat
            except:
                logging.exception("Error in getPanoIDfrmLonlat(): %s", e)
                return 0, 0, 0

        else:  # if there is no panorama
            logging.info("Found no panorama in getPanoIDfrmLonlat(): lon: %f, lat: %f", lon, lat)
            return 0, 0, 0


    def getPanoJsonfrmLonat(self, lon, lat):
        try:
            panoId, pano_lon, pano_lat = self.getPanoIDfrmLonlat(lon, lat)
            jdata = self.getJsonfrmPanoID(panoId)
        except Exception as e:
            # print("Error in getPanoJsonfrmLonat():", e)
            logging.exception("Error in getPanoJsonfrmLonat(): %s", e)
            return None

        return jdata


    def getJsonfrmPanoID(self, panoId, dm=1, saved_path=''):
        url = "https://www.google.com/maps/photometa/v1?authuser=0&hl=en&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1szh-CN!2sus!3m3!1m2!1e2!2s{}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3"
        url = url.format(panoId)


        try:
            resp = requests.get(url, proxies=None)
            line = resp.text.replace(")]}'\n", "")
            try:
                jdata = json.loads(line)
                jdata = utils.compressJson(jdata)
                jdata = utils.refactorJson(jdata)
            except Exception as e:
                jdata = 0  # if there is no panorama
                print("Error in getJsonfrmPanoID(), e, url:", panoId, url, e)
                logging.exception("Error in getJsonfrmPanoID(), %s, %s, %s.", panoId, url, e)
                return jdata

            if saved_path != '':
                self.saved_path = saved_path
                if not os.path.exists(self.saved_path):
                    os.makedirs(self.saved_path)
                try:
                    with open(os.path.join(saved_path, panoId + '.json'), 'w') as f:
                        json.dump(jdata, f)
                except Exception as e:
                    logging.exception("Error in getJsonfrmPanoID() saving json file, %s, %s, %s.", panoId, url, e)

            if dm == 0:
                jdata['model']['depth_map'] = ''
            return jdata

        except Exception as e:
            logging.exception("Error in getJsonfrmPanoID() %s, %s, %s." , panoId, url, e)
            return 0

    def _enlarge_mask(self, arr_mask, width, height, erison_size=2):
        kernel = morphology.disk(erison_size)
        mask = morphology.erosion(arr_mask, kernel).astype(np.int8)
        # mask = Image.fromarray(mask).resize((width, height), Image.LINEAR)
        # resized_mask = np.array(resized_mask)
        resized_mask = Image.fromarray(mask).resize((width, height), Image.LINEAR)
        resized_mask = np.array(resized_mask)
        return resized_mask

    def calculate_xy(self, transformer=None):
        if transformer and (self.lat is not None) and (self.lon is not None ):
            self.x, self.y = transformer.transform(self.lat, selfF.lon)
            return self.x, self.y

        if self.crs_local and (self.lat is not None) and (self.lon is not None ):
            transformer = utils.epsg_transform(in_epsg=4326, out_epsg=self.crs_local)
            self.x, self.y = transformer.transform(self.lat, self.lon)
            return self.x, self.y




    def get_depthmap(self, zoom, saved_path=""):  # return: {}

        if (self.depthmap['depthMap'] is None) or (self.depthmap['zoom'] != zoom):
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None

                depthMapData = utils.parse(self.jdata['model']['depth_map'])


                # parse first bytes to describe data
                header = utils.parseHeader(depthMapData)
                # parse bytes into planes of float values


                data = utils.parsePlanes(header, depthMapData)
                # compute position and values of pixels
                depthmap_dict = utils.computeDepthMap(header, data["indices"], data["planes"])
                depthMap = depthmap_dict['depthMap']

                # move up a row
                # depthMap[:-1] = depthMap[1:]
                # normal_vector_map = depthmap_dict['normal_vector_map']
                # normal_vector_map[:-1] = normal_vector_map[1:]
                # depthmap_dict['normal_vector_map'] = normal_vector_map
                # plane_idx_map = depthmap_dict['plane_idx_map']
                # plane_idx_map[:-1] = plane_idx_map[1:]
                # depthmap_dict['plane_idx_map'] = plane_idx_map


                dm_mask = np.where(np.array(depthMap) > 0, 1, 0).astype(np.int8)

                max_zoom = len(self.jdata['Data']['level_sizes']) - 1
                if zoom > max_zoom:
                    logger.info("%s has no zoom %d depthmap. Used zoom %d instead." % (self.panoId, zoom, max_zoom))
                    zoom = max_zoom
                    # self.p
                    zoom = min(zoom, max_zoom)

                try:
                    image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
                    image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

                except Exception as e:
                    print("Error in get_depthmap() get image_width/height:", self.panoId, e)

                ground_mask = np.where(depthmap_dict['normal_vector_map'][:, :, 2] < GROUND_VECTOR_THRES, 1, 0).astype(np.int8)  # < 10 is correct.


                # resize
                if zoom > 0:
                    self.depthmap['width'] = image_width
                    self.depthmap['height'] = image_height
                    depthMap = Image.fromarray(depthMap).resize((image_width, image_height), Image.LINEAR)
                    dm_mask = self._enlarge_mask(dm_mask, image_width, image_height, erison_size=2)
                    ground_mask = Image.fromarray(ground_mask).resize((image_width, image_height), Image.LINEAR)
                else:
                    self.depthmap['width'] = 512
                    self.depthmap['height'] = 256

                depthMap = np.array(depthMap)


                # dm_mask = np.where(np.array(dm_mask) > 0, 1, 0).astype(int)
                # dm_mask = morphology.erosion(dm_mask, kernel)

                depthMap = depthMap * dm_mask

                # dm_mask = Image.fromarray(dm_mask)
                # dm_mask = dm_mask.resize((image_width, image_height), Image.NEAREST)
                # dm_mask = np.array(dm_mask)

                # depthmap_dict['normal_vector_map'] has been rescaled to [0, 255] from (-1,1).
                # ground_mask[:, 0] = np.where(ground_mask[:,  0] > 100, ground_mask[:, 0], 200)
                # ground_mask[:, 2] = np.where(ground_mask[:,  2] < 10,1, 0)
                #

                ground_mask = np.array(ground_mask)
                ground_mask = ground_mask * dm_mask

                # interpolation using linear regression
                row_num =  image_width
                hc = depthMap[-1][int(image_width/2)]

                # for i in range(row_num):
                #     try:
                #         mask_column = ground_mask[:, i]
                #         depths = depthMap['depthMap'][:, i] * mask_column
                #         y = depths[depths > 0]
                #         # y = y.reshape(-1, 1)
                #         x = np.argwhere(depths > 0).ravel()
                #
                #         f = interpolate.interp1d(x, y)
                #         x_new = np.arange(0, self.depthmap['height'], 1)
                #         y_new = f(x_new)
                #
                #         plt.plot(x, y, 'o', x_new, y_new, '-')
                #         plt.show()
                #
                #         v_resol = math.pi/self.depthmap['height']
                #         thetas =  math.pi/2 - x * v_resol
                #         # tanx = 1/np.tan(thetas) - hc/(y * np.sin(thetas))
                #         tanx = (np.cos(thetas) * y - hc)/(np.sin(thetas) * y)
                #         tanx = np.nan_to_num(tanx)
                #         tanx[tanx >1000] = 0
                #         print("tanx: ", np.arctan(tanx).mean(), np.arctan(tanx).std())
                #
                #         # reg = LinearRegression().fit(x, y)
                #         # print("reg.coef_, reg.intercept_:", reg.coef_, reg.intercept_)
                #         # print("reg score:", reg.score(x, y))
                #     except Exception as e:
                #         logging.exception("Error in get_dempthmap() linear regression: %s", e)
                #

                # if zoom > 0:
                #     im =depthMap

                self.depthmap['depthMap'] = depthMap
                self.depthmap['dm_mask'] = dm_mask
                self.depthmap['ground_mask'] = ground_mask
                self.depthmap['normal_vector_map'] = depthmap_dict['normal_vector_map']
                self.depthmap['plane_idx_map'] = depthmap_dict['plane_idx_map']
                self.depthmap['zoom'] = zoom

                if len(saved_path) > 0:
                    if not os.path.exists(saved_path):
                        os.path.makedirs()
                    new_name = os.path.join(saved_path, self.jdata['Location']['panoId'] + ".tif")
                    # im = depthMap

                    # im[np.where(im == max(im))[0]] = 0

                    # im = im.reshape((depthMap["height"], depthMap["width"]))  # .astype(int)
                    # display image
                    img = PIL.Image.fromarray(depthMap)
                    img.save(new_name)
                return self.depthmap
            except Exception as e:
                logger.exception("Error in get_depthmap(): %s", e)

        return self.depthmap

    def get_ground_points(self, zoom=0, color=False, img_type='pano'):
        '''

        :param zoom:
        :param color:
        :param img_type:  'pano' or 'seg'
        :return:
        '''

        depthmap = self.get_depthmap(zoom=zoom)['depthMap']
        ground_mask = self.get_depthmap(zoom=zoom)['ground_mask']
        depthmap = depthmap * ground_mask
        arr_rowcol = np.argwhere(depthmap > 0)

        arr_col = arr_rowcol[:, 1]
        arr_row = arr_rowcol[:, 0]

        ground_points = self.col_row_to_points(arr_col, arr_row, zoom=zoom)
        if color:
            colors = self.get_pixel_from_row_col(arr_col, arr_row, zoom=zoom, img_type=img_type)
            ground_points = np.concatenate([ground_points, colors], axis=1)

        return ground_points

    def calculate_DEM(self, ground_points, width=50, height=50, resolution=0.05, dem_coarse_resolution = 0.4, smooth_sigma=DEM_SMOOTH_SIGMA):
        P = ground_points
        P = P[P[:, 0] < width/2]
        P = P[P[:, 0] > -width/2]
        P = P[P[:, 1] < height/2]
        P = P[P[:, 1] > -height/2]

        # convert to grid coordinates systems
        P_col = (P[:, 0]/dem_coarse_resolution + int(width / dem_coarse_resolution / 2)).astype(int)
        P_row = (int(height / dem_coarse_resolution / 2) - P[:, 1] / dem_coarse_resolution).astype(int)

        dem_coarse_np = np.ones((int(height / dem_coarse_resolution), int(width / dem_coarse_resolution))) * -999
        dem_coarse_np[P_row, P_col] = P[:, 2]

        # grid_col = np.linspace(-width/2,  width/2,  dem_coarse_np.shape[1])
        # grid_row = np.linspace(height/2, -height/2, dem_coarse_np.shape[0])

        # resolution = 0.03
        w = int(width / resolution)
        h = int(height / resolution)
        # dem_fined = np.array(Image.fromarray(dem_coarse_np).resize((w, h), Image.LINEAR))

        grid_col = np.linspace(-width/2,  width/2, w)
        grid_row = np.linspace(height/2, -height/2, h)

        # make a Kriging interpolation model
        # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/examples/00_ordinary.html#sphx-glr-examples-00-ordinary-py

        idx = np.argwhere(dem_coarse_np > -100)
        z = dem_coarse_np[dem_coarse_np > -100]

        # mask = np.where(dem_coarse_np > -100, 1, 0)

        cols = dem_coarse_np.shape[1]
        rows = dem_coarse_np.shape[0]
        xx, yy = idx[:,1].astype(float), idx[:,0].astype(float)

        OK = OrdinaryKriging(
        xx,
        yy,
        z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        )

        empty_idx = np.argwhere(dem_coarse_np < -100)

        z, ss = OK.execute(style="points",    # "mask" is not good and slow (*1.5 times)
                           xpoints=empty_idx[:, 1].astype(float),   # np.arange(0.0, cols, 1.0),
                           ypoints=empty_idx[:, 0].astype(float),   # np.arange(0.0, rows, 1.0),
                           # mask = mask,
                           backend='vectorized')  # backend: "C" is very slow (X 7 times)
        # https://pykrige.readthedocs.io/en/latest/generated/pykrige.ok.OrdinaryKriging.html

        # dem_coarse_np = z
        dem_coarse_np[empty_idx[:, 0], empty_idx[:, 1]] = z
        dem_coarse_np = gaussian_filter(dem_coarse_np, sigma=smooth_sigma)

        dem_refined = Image.fromarray(dem_coarse_np).resize((int(width/resolution), int(height/resolution)), Image.LINEAR)
        dem_refined = np.array(dem_refined)



    # get the camera height
        central_row = int(dem_refined.shape[0] / 2)
        central_col = int(dem_refined.shape[1] / 2)
        camera_height = -dem_refined[central_row, central_col]
        elevation_wgs84_m = self.jdata['Location']['elevation_wgs84_m']
        elevation_egm96_m = self.jdata['Location']['elevation_egm96_m']

        dem_refined = dem_refined + elevation_egm96_m # + camera_height

        transformer = utils.epsg_transform(4326, self.crs_local)  # New Jersey state plane, meter
        x_m, y_m = transformer.transform(self.lat, self.lon)




        self.DEM['DEM'] = dem_refined

        self.DEM['resolution'] = resolution
        self.DEM['central_x'] = x_m
        self.DEM['central_y'] = y_m
        self.DEM['camera_height'] = camera_height

        return self.DEM

    def points_to_DOM(self, Xs, Ys, pixels, resolution):
        try:
            minX = min(Xs)
            maxY = max(Ys)
            if len(Xs) != len(Ys):
                print("len(Xs) != len(Ys)")
                return None, None

            if len(pixels) != len(Ys):
                print("len(pixels) != len(Ys)")
                return None, None

            rangeX = max(Xs) - minX
            rangeY = maxY - min(Ys)
            w = int(rangeX / resolution)
            h = int(rangeY / resolution)
            channels = len(pixels.shape)
            if channels == 2:
                channels = pixels.shape[1]

            cols = ((Xs - minX) / resolution)
            cols = cols.astype(int)
            cols = np.where(cols > w - 1, w - 1, cols)
            rows = (maxY - Ys) / resolution
            rows = rows.astype(int)
            rows = np.where(rows > h - 1, h - 1, rows)


            if channels == 1:
                np_image = np.zeros((h, w))
            else:
                np_image = np.zeros((h, w, channels))

            transformer = utils.epsg_transform(4326, self.crs_local)  # New Jersey state plane, meter
            x_m, y_m = transformer.transform(self.lat, self.lon)

            np_image[rows, cols] = pixels

            worldfile = [resolution, 0, 0, -resolution, x_m - rangeX / 2, y_m + rangeY / 2]

            return np_image, worldfile
        except Exception as e:
            print("Error in points_to_DOM():", e)

    def get_road_plane(self, resolution=0.1, width=40, height=40):
        img_w = int(width / resolution * 1.7)
        img_h = int(height / resolution * 1.7)

        tilt_yaw_deg = self.jdata['Projection']['tilt_yaw_deg']
        pano_yaw_deg = self.jdata['Projection']['pano_yaw_deg']
        tilt_pitch_deg = self.jdata['Projection']['tilt_pitch_deg']
        elevation_egm96_m = self.jdata['Location']['elevation_egm96_m']

        grid_col = np.linspace(-width/2,  width/2, img_w)
        grid_row = np.linspace(height/2, -height/2, img_h)
        Xs, Ys = np.meshgrid(grid_col, grid_row)

        rotate_x_radian = math.radians(90 - tilt_yaw_deg)
        rotate_y_radian = math.radians(
        -tilt_pitch_deg)  # should  be negative according to the observation of highway ramp. ???
        # rotate_z_radian = math.radians(90 - pano_yaw_deg)
        rotate_z_radian = math.radians(pano_yaw_deg)
        self.get_DEM(zoom=0)
        camera_height = self.DEM['camera_height'] - elevation_egm96_m

        zoom = 3

        np_plane = np.ones((img_h, img_w)) * camera_height
        P = np.concatenate([Xs.ravel().reshape(-1, 1), Ys.ravel().reshape(-1, 1), np_plane[np_plane > -999].reshape((-1, 1))], axis=1)
        P = P.dot(utils.rotate_x(rotate_x_radian))  # math.radians(90 - tilt_yaw_deg)  # roll
        P = P.dot(utils.rotate_y(rotate_y_radian))  # math.radians(-tilt_pitch_deg)  # pitch
        P = P.dot(utils.rotate_z(rotate_z_radian))  # math.radians(90 - pano_yaw_deg)  # yaw

        thetas, phis = self.XYZ_to_spherical(P)  # input：meters
        colors = self.find_pixel_to_thetaphi(thetas, phis, zoom=zoom, img_type="DOM")

        P = np.concatenate([P, colors], axis=1)



    # P[:, 2] = P[:, 2] + elevation_egm96_m


        max_slope = math.radians(15)
        w_temp = img_w

        return P

    # unfinished............
    def get_DEM(self, width=40, height=40, resolution=0.4, dem_coarse_resolution=DEM_COARSE_RESOLUTION, zoom=1, smooth_sigma=0):  # return: numpy array,

        new_name = os.path.join(self.saved_path, self.panoId + f"_DEM_{resolution:.2f}.tif")
        worldfile_name = new_name.replace(".tif", ".tfw")

        if os.path.exists(new_name):

            # self.DEM['colors'] = colors
            self.DEM['resolution'] = resolution
            transformer = utils.epsg_transform(4326, self.crs_local)  # New Jersey state plane, meter
            elevation_egm96_m = self.jdata['Location']['elevation_egm96_m']
            self.DEM['DEM'] = np.array(Image.open(new_name)) #  - elevation_egm96_m

            x_m, y_m = transformer.transform(self.lat, self.lon)
            self.DEM['central_x'] = x_m
            self.DEM['central_y'] = y_m
            # get the camera height
            central_row = int(self.DEM['DEM'].shape[0] / 2)
            central_col = int(self.DEM['DEM'].shape[1] / 2)
            camera_height = self.DEM['DEM'][central_row, central_col]
            self.DEM['camera_height'] = camera_height
            self.DEM['zoom'] = zoom

            return self.DEM

        if (self.DEM['DEM'] is None) or (self.DEM['resolution'] != resolution):
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None
                ground_points = self.get_ground_points(zoom=zoom)



                # xx_coarse = coarse_idx[:, 0] * dem_coarse_resolution - width/2
                # yy_coarse = height/2 - coarse_idx[:, 1] * dem_coarse_resolution

                # interpolate_fun = interpolate.interp2d(xx_coarse, yy_coarse, coarse_values, kind='linear', copy=False)
                #
                # xx = np.arange(-width/2, width/2, resolution)
                # yy = np.arange(height/2, -height/2, -resolution)  # easy to make a mistake.
                # xxx, yyy = np.meshgrid(xx, yy)
                #
                # znew = interpolate_fun(xx, yy)
                #
                # print("znew:", znew.shape)
                # print("znew max, min in Z axis:")
                # print(np.max(znew))
                # print(np.min(znew))

                DEM = self.calculate_DEM(ground_points=ground_points,
                                         width=width,
                                         height=height,
                                         resolution=resolution,
                                         dem_coarse_resolution=dem_coarse_resolution,
                                         smooth_sigma=smooth_sigma)


                # save DEM
                if self.saved_path != "":
                    try:
                        if not os.path.exists(self.saved_path):
                            os.mkdir(self.saved_path)
                            # im = Image.fromarray(self.DEM['DEM'])
                        im = Image.fromarray(DEM['DEM'])
                        worldfile_name = new_name.replace(".tif", ".tfw")
                        worldfile = [resolution,
                                     0,
                                     0,
                                     -resolution,
                                     DEM['central_x'] - width/2,
                                     DEM['central_y'] + height/2]
                        # im.show()
                        im.save(new_name)

                        with open(worldfile_name, 'w') as wf:
                            for line in worldfile:
                                # print(line)
                                wf.write(str(line) + '\n')
                    except Exception as e:
                        print("Error in saving DEM: ", self.saved_path, e)

            except Exception as e:
                logger.exception("Error in get_DEM(): %s", e)

        return self.DEM

    def XYZ_to_spherical(self, XYZs):  # no rotation.
        # panorama = self.get_panorama(zoom=5)['image']
        # image_height, image_width, channel = panorama.shape
        # nx, ny = (image_width, image_height)
        # x_space = np.linspace(0, nx - 1, nx)
        # y_space = np.linspace(ny - 1, 0, ny)
        #
        # xv, yv = np.meshgrid(x_space, y_space)
        #
        # thetas = yv / ny * np.pi - np.pi / 2
        # phis = xv / nx * (np.pi * 2) - np.pi
        # thetas_sin = np.sin(thetas)
        # thetas_cos = np.cos(thetas)
        # phis_sin = np.sin(phis)
        # phis_cos = np.cos(phis)

        tilt_yaw_deg = self.jdata['Projection']['tilt_yaw_deg']
        pano_yaw_deg = self.jdata['Projection']['pano_yaw_deg']
        tilt_pitch_deg = self.jdata['Projection']['tilt_pitch_deg']

        rotate_x_radian = math.radians(90 - tilt_yaw_deg)
        rotate_y_radian = math.radians(
            -tilt_pitch_deg)  # should  be negative according to the observation of highway ramp. ???
        # rotate_z_radian = math.radians(90 - pano_yaw_deg)
        rotate_z_radian = math.radians(pano_yaw_deg)

        # P0 = XYZs
        XYZs = XYZs[:, :3]
        XYZs = XYZs.dot(utils.rotate_z(-rotate_z_radian))
        XYZs = XYZs.dot(utils.rotate_y(-rotate_y_radian))
        XYZs = XYZs.dot(utils.rotate_x(-rotate_x_radian))

        X = XYZs[:, 0]
        Y = XYZs[:, 1]
        Z = XYZs[:, 2] -  self.jdata['Location']['elevation_egm96_m']

        R = np.sqrt(X**2 + Y**2)

        dm_np = np.sqrt(X**2 + Y**2 + Z**2)

        theta_sin = Z / dm_np
        phi_sin = X / R

        theta = np.arcsin(theta_sin)
        # phi   = math.pi/2 - np.arcsin(phi_sin) -math.pi
        # phi   =  np.arcsin(phi_sin)
        phi = np.arctan2(X, Y)


        theta = np.nan_to_num(theta)
        phi = np.nan_to_num(phi)


        # find the pixels
        # v_reso = math.pi/image_height
        # h_reso = math.pi * 2 / image_width
        # row = ((theta - math.pi/2 ) / -v_reso).astype(int)
        # col = ((phi + math.pi) / h_reso).astype(int)
        # row[row >= image_height] = image_height - 1
        # col[col >= image_width] = image_width - 1
        #
        # new_img = np.zeros((image_height, image_width, channel)).astype(int)
        # new_img[row, col] = panorama[row, col]
        # im = Image.fromarray(new_img.astype('uint8'), 'RGB')
        # im.show()

        return theta, phi

    def get_segmentation(self, zoom=4, fill_clipped_seg=False):

        if not os.path.exists(self.segmenation['full_path']):
            logger.error("No full_path for segmentation file! ", self.panoId, exc_info=True)
            return None


        if self.segmenation['segmentation'] is None:
            img_pil = Image.open(self.segmenation['full_path'])
            # img_pil = img_pil.convert('RGB')

            # fill the clipped segmentation:

            # fill_clipped_seg = True
            if fill_clipped_seg:
                w, h = img_pil.size
                large_img = Image.new('P', (w * 1, h * 4))  # new image
                large_img.paste(img_pil, (0, h*2))
                draw = ImageDraw.Draw(large_img)
                draw.rectangle([0, h * 3, w, h * 4], fill=10, width=0)
                img_pil = large_img
                # img_pil.show()

            self.segmenation['segmentation'] = np.array(img_pil)

        return self.segmenation


    def set_segmentation_path(self, full_path):
        self.segmenation['full_path'] = full_path


    def get_pixel_from_row_col(self, arr_col, arr_row, zoom=4, img_type="pano", fill_clipped_seg=False):
        '''

        :param arr_row:
        :param arr_col:
        :param zoom:
        :param img_type: "pano" or "seg"
        :return:
        '''
        np_img = None
        if img_type == "pano":
            np_img = self.get_panorama(zoom=zoom)['image']
        if img_type == "seg":
            np_img = self.get_segmentation(zoom=zoom, fill_clipped_seg=fill_clipped_seg)['segmentation']

        if len(np_img.shape) > 2:
            image_height, image_width, channel = np_img.shape
        if len(np_img.shape) == 2:
            image_height, image_width = np_img.shape

        return np_img[arr_row, arr_col]




    def find_pixel_to_thetaphi(self, theta, phi, zoom=4, img_type="DOM", fill_clipped_seg=False):
        ''':argument
        theata, phi: numpy array
        type: DOM or segmentation
        '''
        # panorama = None
        if img_type == "DOM":
            panorama = self.get_panorama(zoom=zoom)['image']

        if img_type == "segmentation":
            panorama = self.get_segmentation(zoom=zoom, fill_clipped_seg=fill_clipped_seg)['segmentation']

        if len(panorama.shape) > 2:
            image_height, image_width, channel = panorama.shape
        if len(panorama.shape) == 2:
            image_height, image_width = panorama.shape
            # palette = np.array(im.getpalette(),dtype=np.uint8).reshape((256,3))
            channel = 1

        v_reso = math.pi/image_height
        h_reso = math.pi * 2 / image_width
        row = ((theta - math.pi/2 ) / -v_reso).astype(int)
        col = ((phi + math.pi) / h_reso).astype(int)
        row[row >= image_height] = image_height - 1
        col[col >= image_width] = image_width - 1

        # new_img = np.zeros((image_height, image_width, channel)).astype(int)
        # new_img[row, col] = panorama[row, col]
        # im = Image.fromarray(new_img.astype('uint8'), 'RGB')
        # im.show()

        return panorama[row, col]

    def calculate_DOM(self, width = 40, height = 40, resolution=0.03, zoom=4, img_type="DOM", fill_clipped_seg=False):
        w = int(width / resolution)
        h = int(height / resolution)

        grid_col = np.linspace(-width/2,  width/2, w)
        grid_row = np.linspace(height/2, -height/2, h)

        DEM = self.get_DEM(width=width,
                           height=height,
                           resolution=resolution,
                           dem_coarse_resolution=DEM_COARSE_RESOLUTION,
                           zoom=zoom)

        Xs, Ys = np.meshgrid(grid_col, grid_row)
        Zs = DEM['DEM']  # - self.jdata['Location']['elevation_egm96_m']
        XYZs = np.concatenate(
            [Xs.ravel().reshape(-1, 1), Ys.ravel().reshape(-1, 1), Zs.ravel().reshape(-1, 1)],
            axis=1)

        # XYZs = XYZs # + self.DEM['camera_height']


        thetas, phis = self.XYZ_to_spherical(XYZs)  # input：meters

        colors = self.find_pixel_to_thetaphi(thetas, phis, zoom=zoom, img_type=img_type, fill_clipped_seg=fill_clipped_seg)
        channels = int(colors.size / w / h)
        if channels > 2:
            np_DOM = colors.reshape((h, w, 3))
        if channels == 1:
            np_DOM = colors.reshape((h, w))

        self.DOM['DOM'] = np_DOM
        # XYZs[:, 2] = XYZs[:, 2] + self.jdata['Location']['elevation_egm96_m']
        # self.DOM['DOM_points'] = np.concatenate([XYZs, colors], axis=1)
        # self.DOM['DOM_points'] = self.get_DOM_points(width=width, height=height, resolution=resolution,
        #                                              zoom=zoom, img_type=img_type,fill_clipped_seg=fill_clipped_seg)

        return self.DOM

    def get_DOM_points(self, width = 40, height = 40, resolution=0.03, zoom=3, img_type="DOM",fill_clipped_seg=False):
        w = int(width / resolution)
        h = int(height / resolution)

        grid_col = np.linspace(-width/2,  width/2, w)
        grid_row = np.linspace(height/2, -height/2, h)
        DEM = self.get_DEM(width = width, height = height,
                           resolution=resolution,
                           zoom=zoom,
                           dem_coarse_resolution=DEM_COARSE_RESOLUTION,
                           smooth_sigma=DEM_SMOOTH_SIGMA)
        DOM = self.get_DOM(width = width, height = height, resolution=resolution, zoom=zoom, img_type=img_type,fill_clipped_seg=fill_clipped_seg)

        Xs, Ys = np.meshgrid(grid_col, grid_row)
        Zs = DEM['DEM']  # - self.jdata['Location']['elevation_egm96_m']
        XYZs = np.concatenate([Xs.ravel().reshape(-1, 1),
                              Ys.ravel().reshape(-1, 1),
                              Zs.ravel().reshape(-1, 1)],
                              axis=1)

        colors = DOM['DOM'].reshape((-1, 3))
        DOM_points = np.concatenate([XYZs, colors], axis=1)
        self.DOM['DOM_points'] = DOM_points

        return DOM_points

    def get_DOM(self, width = 40, height = 40, resolution=0.03, zoom=3, img_type="DOM",fill_clipped_seg=False):  # return: numpy array,
        """
        :param width:
        :param height:
        :param resolution:
        :param zoom:
        :param type: DOM or segmentation
        :return:
        """
        new_name = os.path.join(self.saved_path, self.panoId + f"_DOM_{resolution:.2f}.tif")
        worldfile_name = new_name.replace(".tif", ".tfw")
        self.DOM['resolution'] = resolution
        transformer = utils.epsg_transform(4326, self.crs_local)  # New Jersey state plane, meter

        x_m, y_m = transformer.transform(self.lat, self.lon)
        self.DOM['central_x'] = x_m
        self.DOM['central_y'] = y_m

        if os.path.exists(new_name):
            self.DOM['DOM'] = np.array(Image.open(new_name))
            # self.DOM["DOM_points"] = self.get_DOM_points(width=width,
            #                                              height=height,
            #                                              resolution=resolution,
            #                                              zoom=zoom,
            #                                              img_type=img_type,
            #                                              fill_clipped_seg=fill_clipped_seg)
            # self.DEM['colors'] = colors

            # get the camera height
            # central_row = int(self.DOM['DOM'].shape[0] / 2)
            # central_col = int(self.DOM['DOM'].shape[1] / 2)
            # camera_height = self.DOM['DOM'][central_row, central_col]

            return self.DOM

        if (self.DOM['DOM'] is None) or (self.DOM['resolution'] != resolution):
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None

                # get the DOM
                DOM = self.calculate_DOM(width = width,
                                          height = height,
                                          resolution=resolution,
                                          zoom=4,
                                          img_type=img_type,
                                          fill_clipped_seg=fill_clipped_seg
                                          )

                if self.saved_path != "":
                        if not os.path.exists(self.saved_path):
                            os.mkdir(self.saved_path)
                        # im = Image.fromarray(self.DEM['DEM'])
                        channels = 1
                        if len(DOM['DOM'].shape) > 2:
                            channels = DOM['DOM'].shape[2]
                        if channels == 3:
                            im = Image.fromarray(DOM['DOM'].astype("uint8"), "RGB")

                        if channels == 1:
                            im = Image.fromarray(DOM['DOM'].astype("uint8"), "P")
                            try:
                                palette = Image.open(self.segmenation['full_path']).getpalette()
                                im.putpalette(palette)
                            except Exception as e:
                                print("Error in Image.putpalette():", e)

                        new_name = os.path.join(self.saved_path, self.panoId + f"_DOM_{resolution:.2f}.tif")
                        worldfile_name = new_name.replace(".tif", ".tfw")
                        worldfile = [resolution, 0, 0, -resolution, x_m - width/2, y_m + height/2]
                        # im.show()
                        im.save(new_name)

                        with open(worldfile_name, 'w') as wf:
                            for line in worldfile:
                                # print(line)
                                wf.write(str(line) + '\n')

                return self.DOM

            except Exception as e:
                logger.exception("Error in get_DOM(): %s", e)



    def col_row_to_angles(self, arr_col, arr_row, zoom=4):
        '''
        Convert the pixels (rows, cols) to spherical coordinates (theta, phi).
        :param arr_col:
        :param arr_row:
        :param zoom:
        :return: coordinate of pixel center
        '''
        image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
        image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

        thetas = np.pi / 2 - (arr_row ) / image_height * np.pi
        phis = (arr_col ) / image_width * (np.pi * 2) - np.pi
        return thetas, phis



    def col_row_to_points(self, arr_col, arr_row, zoom=4):
        '''
        Convert theta/phi angles to points.
        :param theta_phi_list:
        :param zoom:
        :return:
        '''
        depthmap = self.get_depthmap(zoom=zoom)['depthMap']
        image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
        image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

        self.point_cloud["dm_mask"] = depthmap


        tilt_yaw_deg = self.jdata['Projection']['tilt_yaw_deg']
        pano_yaw_deg = self.jdata['Projection']['pano_yaw_deg']
        tilt_pitch_deg = self.jdata['Projection']['tilt_pitch_deg']
        elevation_egm96_m = self.jdata['Location']['elevation_egm96_m']

        # distances = depthmap[arr_row.astype(int), (arr_col).astype(int)]
        distances = depthmap[np.ceil(arr_row).astype(int), (arr_col).astype(int)]


        arr_theta, arr_phi = self.col_row_to_angles(arr_col, arr_row, zoom=zoom)

        thetas_sin = np.sin(arr_theta)
        thetas_cos = np.cos(arr_theta)
        phis_sin = np.sin(arr_phi)
        phis_cos = np.cos(arr_phi)

        R = thetas_cos * distances

        z = distances * thetas_sin

        y = R * phis_cos
        x = R * phis_sin
        P = np.concatenate([x.ravel().reshape(-1, 1), y.ravel().reshape(-1, 1), z.ravel().reshape(-1, 1)],
                           axis=1)

        rotate_x_radian = math.radians(90 - tilt_yaw_deg)
        rotate_y_radian = math.radians(
            -tilt_pitch_deg)  # should  be negative according to the observation of highway ramp. ???
        # rotate_z_radian = math.radians(90 - pano_yaw_deg)
        rotate_z_radian = math.radians(pano_yaw_deg)

        pitch = self.jdata['Projection']['tilt_pitch_deg']
        theta = self.jdata['Projection']['tilt_yaw_deg']
        phi = math.radians(90 - pano_yaw_deg)
        pitch = math.radians(pitch)
        theta = math.radians(theta)
        theta = (theta - pi / 2)
        #
        # m = np.eye(3)
        # m = m.dot(utils.rotate_z(pitch))
        # m = m.dot(utils.rotate_x(theta))
        # # m = m.dot(utils.rotate_y(phi))
        #
        # P = P.dot(m.T)

        #
        # P = P.dot(utils.rotate_x(pitch))  # math.radians(-tilt_pitch_deg)  # pitch
        # P = P.dot(utils.rotate_y(theta))  # math.radians(90 - tilt_yaw_deg)  # roll
        # P = P.dot(utils.rotate_z(phi))  # math.radians(90 - pano_yaw_deg)  # yaw


        P = P.dot(utils.rotate_x(rotate_x_radian))  # math.radians(90 - tilt_yaw_deg)  # roll
        P = P.dot(utils.rotate_y(rotate_y_radian))  # math.radians(-tilt_pitch_deg)  # pitch
        P = P.dot(utils.rotate_z(rotate_z_radian))  # math.radians(90 - pano_yaw_deg)  # yaw


        # P = np.concatenate([P, dm_np.ravel().reshape(-1, 1), thetas.reshape(-1,1), phis.reshape(-1, 1)], axis=1)
        P = np.concatenate([P, distances.reshape(-1, 1)], axis=1)


        # P = P[np.where(dm_mask.ravel())]
        # distance_threshole = 20
        # P = P[P[:, 3] < distance_threshole]
        # P = P[P[:, 3] > 0]

        return P


    def get_point_cloud(self, zoom=0, distance_threshole=40, color=True, saved_path=""):  # return: numpy array
        ''':param
        pano_zoom: -1: not colorize the point cloud
        '''

        if (self.point_cloud['point_cloud'] is None) or (self.point_cloud['zoom'] != zoom):
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None

                depthmap = self.get_depthmap(zoom=zoom)['depthMap']


                image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
                image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

                dm_resized = Image.fromarray(depthmap).resize((image_width, image_height), Image.BICUBIC)
                kernel = morphology.disk(2)
                dm_mask = morphology.erosion(depthmap, kernel)
                dm_mask = np.where(np.array(dm_mask) > 0, 1, 0).astype(int)
                dm_mask = Image.fromarray(dm_mask)
                dm_mask = dm_mask.resize((image_width, image_height), Image.NEAREST)
                dm_mask = np.array(dm_mask)
                self.point_cloud["dm_mask"] = dm_mask
                # dm_mask[0, :] = 1  # show the camera center.

                tilt_yaw_deg = self.jdata['Projection']['tilt_yaw_deg']
                pano_yaw_deg = self.jdata['Projection']['pano_yaw_deg']
                tilt_pitch_deg = self.jdata['Projection']['tilt_pitch_deg']
                elevation_egm96_m = self.jdata['Location']['elevation_egm96_m']

                dm_np = np.array(dm_resized).astype(float)
                dm_np = dm_np * dm_mask

                img = self.get_panorama(zoom=zoom)['image']

                img_np = np.array(img)[0:image_height, :, :]
                colors = img_np.reshape(-1, 3)

                normalvector = Image.fromarray(self.depthmap['normal_vector_map']).resize((image_width, image_height), Image.NEAREST)
                normalvector_np = np.array(normalvector)
                normalvectors = normalvector_np.reshape(-1, 3)

                plane_idx = Image.fromarray(self.depthmap['plane_idx_map']).resize((image_width, image_height), Image.NEAREST)
                plane_np = np.array(plane_idx)
                planes = plane_np.reshape(-1, 1)

                nx, ny = (image_width, image_height)
                x_space = np.linspace(0, nx - 1, nx)
                y_space = np.linspace(ny - 1, 0, ny)

                xv, yv = np.meshgrid(x_space, y_space)

                thetas = yv / ny * np.pi - np.pi / 2
                phis = xv / nx * (np.pi * 2) - np.pi
                thetas_sin = np.sin(thetas)
                thetas_cos = np.cos(thetas)
                phis_sin = np.sin(phis)
                phis_cos = np.cos(phis)
                R = thetas_cos * dm_np  # * dm_mask

                z = dm_np * thetas_sin
                # x = R * phis_cos
                # y = R * phis_sin
                y = R * phis_cos
                x = R * phis_sin
                P = np.concatenate([x.ravel().reshape(-1, 1), y.ravel().reshape(-1, 1), z.ravel().reshape(-1, 1)],
                                   axis=1)

                rotate_x_radian = math.radians(90 - tilt_yaw_deg)
                rotate_y_radian = math.radians(
                    -tilt_pitch_deg)  # should  be negative according to the observation of highway ramp. ???
                # rotate_z_radian = math.radians(90 - pano_yaw_deg)
                rotate_z_radian = math.radians(pano_yaw_deg)

                P = P.dot(utils.rotate_x(rotate_x_radian))  # math.radians(90 - tilt_yaw_deg)  # roll
                P = P.dot(utils.rotate_y(rotate_y_radian))  # math.radians(-tilt_pitch_deg)  # pitch
                P = P.dot(utils.rotate_z(rotate_z_radian))  # math.radians(90 - pano_yaw_deg)  # yaw


                #
                # P1 =     P.dot(utils.rotate_x(rotate_x_radian))  # math.radians(90 - tilt_yaw_deg)  # roll
                #
                # P2 = P1.dot(utils.rotate_y(rotate_y_radian))  # math.radians(-tilt_pitch_deg)  # pitch
                # P3 = P2.dot(utils.rotate_z(rotate_z_radian))  # math.radians(90 - pano_yaw_deg)  # yaw
                #
                # P2_re =    P3.dot(utils.rotate_z(-rotate_z_radian))
                # P1_re = P2_re.dot(utils.rotate_y(-rotate_y_radian))
                # P_re  = P1_re.dot(utils.rotate_x(-rotate_x_radian))



                # P = np.concatenate([P, dm_np.ravel().reshape(-1, 1), thetas.reshape(-1,1), phis.reshape(-1, 1)], axis=1)
                P = np.concatenate([P, dm_np.ravel().reshape(-1, 1)], axis=1)

                if color:
                    P = np.concatenate([P, colors, planes, normalvectors], axis=1)

                P = P[np.where(dm_mask.ravel())]
                # distance_threshole = 20
                P = P[P[:, 3] < distance_threshole]
                P = P[P[:, 3] > 0]



                # P = P[:, :6]
                self.point_cloud['point_cloud'] = P
                self.point_cloud['zoom'] = int(zoom)

                return self.point_cloud

            except Exception as e:
                logger.exception("Error in get_depthmap(): %s", e)

    def download_panorama(self, zoom: int=3):
        """Reference:
                    https://developers.google.com/maps/documentation/javascript/streetview
                    See the part from "Providing Custom Street View Panoramas" section.
                    Get those tiles and mosaic them to a large image.
                    The url of a tile:
                    https://geo2.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&panoid=CJ31ttcx7ez9qcWzoygVqA&output=tile&x=1&y=1&zoom=4&nbt&fover=2
                    Make sure randomly use geo0 - geo3 server.
                    When zoom=4, a panorama image have 6 rows, 13 cols.
                """
        try:
            if (str(self.panoId) == str(0)) or (len(self.panoId) < 20):
                logger.info("%s is not a panoId. Returned None", self.panoId)
                return None

            tile_width = self.jdata['Data']['tile_width']
            tile_height = self.jdata['Data']['tile_height']

            zoom = int(zoom)
            image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
            image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

            # passed
            column_cnt = np.ceil(image_width / tile_width).astype(int)
            row_cnt = np.ceil(image_height / tile_height).astype(int)

            target = Image.new('RGB', (tile_width * column_cnt, tile_height * row_cnt))  # new image

            for x in range(column_cnt):  # col
                for y in range(row_cnt):  # row
                    num = random.randint(0, 3)
                    zoom = str(zoom)
                    # example:
                    #  https://geo2.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&panoid=KoQv4Ob8WNJ709enNrQCBQ
                    # &output=tile&x=20&y=6&zoom=5&nbt&fover=2
                    url = 'https://geo' + str(
                        num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&panoid=' + self.panoId + '&output=tile&x=' + str(
                        x) + '&y=' + str(y) + '&zoom=' + zoom + '&nbt&fover=2'
                    file = urllib.request.urlopen(url)
                    image = Image.open(file)
                    if image.size != (tile_width, tile_width):
                        image = image.resize((tile_width, tile_height))
                    target.paste(image,
                                 (tile_width * x, tile_height * y, tile_width * (x + 1), tile_height * (y + 1)))

            # if int(zoom) == 0:
            #     target = target.crop((0, 0, image_width, image_height))
            if target.size != (image_width, image_height):
                target = target.crop((0, 0, image_width, image_height))

            return target

        except Exception as e:
            logger.exception("Error in get_panorama(): %s", e)
            return None

    def get_panorama(self, prefix="", suffix="", zoom: int = 5, check_size=False, skip_exist=True):
        """Reference:
            https://developers.google.com/maps/documentation/javascript/streetview
            See the part from "Providing Custom Street View Panoramas" section.
            Get those tiles and mosaic them to a large image.
            The url of a tile:
            https://geo2.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&panoid=CJ31ttcx7ez9qcWzoygVqA&output=tile&x=1&y=1&zoom=4&nbt&fover=2
            Make sure randomly use geo0 - geo3 server.
            When zoom=4, a panorama image have 6 rows, 13 cols.
        """
        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix

        target = None

        try:
            if (str(self.panoId) == str(0)) or (len(self.panoId) < 20):
                logger.info("%s is not a panoId. Returned None", self.panoId)
                return None

            if (self.panorama['image'] is None) or (zoom != self.panorama['zoom']): # need to download
                tile_width = self.jdata['Data']['tile_width']
                tile_height = self.jdata['Data']['tile_height']

                zoom = int(zoom)
                max_zoom = len(self.jdata['Data']['level_sizes']) - 1
                if zoom > max_zoom:
                    logger.info("%s has no zoom %d panorama. Used zoom %d instead." % (self.panoId, zoom, max_zoom))
                    zoom = max_zoom
                    # self.p
                # zoom = min(zoom, max_zoom)
                image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
                image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

                new_name = os.path.join(self.saved_path, (prefix + self.panoId + suffix + f'_{zoom}.jpg'))

                # if a fined image exists:
                max_zoom = 5
                for img_zoom in range(zoom, max_zoom + 1, 1):
                    jpg_name = os.path.join(self.saved_path, (prefix + self.panoId + suffix + f'_{img_zoom}.jpg'))
                    if os.path.exists(jpg_name):
                        new_name = jpg_name

                if os.path.exists(new_name):

                    old_img = Image.open(new_name)
                    img_zoom = int(new_name[-5])

                    if img_zoom > zoom:
                        old_img = old_img.resize((image_width, image_height))
                        print(f"Resize zoom-leve {img_zoom} to zoom-level {zoom}: {new_name}")

                    if old_img.size == (image_width, image_height):  # no need to download new image
                        target = old_img
                        logger.info("Found existing panorama: %s", new_name)
                        self.panorama["image"] = np.array(target)
                        self.panorama['zoom'] = int(zoom)
                        return self.panorama
                    # old_img.close()

                    else:
                        pass
                else:
                    target = self.download_panorama(zoom=zoom)

                    if self.saved_path != "":
                        if not os.path.exists(self.saved_path):
                            os.mkdir(self.saved_path)
                        # print("new_name:", new_name)
                        target.save(new_name)

                    self.panorama["image"] = np.array(target)
                    self.panorama['zoom'] = int(zoom)
                    return self.panorama

            else:
                return self.panorama

        except Exception as e:
            logger.exception("Error in get_panorama(): %s, %s " % (e,  self.panoId))
            return None



    def clip_pano(self, to_theta=0, to_phi=0, width=1024, height=768, fov_h_deg=90, zoom=5, img_type="pano", saved_path=os.getcwd()):
        if img_type == "pano":
            img = self.get_panorama(zoom=zoom)['image']
        if img_type == "depthmap":
            img = self.get_depthmap(zoom=zoom)['depthMap']

        self.saved_path = saved_path


        pitch = self.jdata['Projection']['tilt_pitch_deg']
        theta = self.jdata['Projection']['tilt_yaw_deg']
        pitch = math.radians(pitch)
        theta = math.radians(theta)
        to_theta = math.radians(to_theta)
        theta = (theta - pi / 2 )

        # to_theta = math.radians(to_theta)
        to_phi = math.radians(to_phi)

        # rotation matrix
        m = np.eye(3)

        # sphere orientation
        m = m.dot(utils.rotate_z(pitch))
        m = m.dot(utils.rotate_x(theta))

        # orientate to the perspective direction
        m = m.dot(utils.rotate_y(to_phi))
        m = m.dot(utils.rotate_x(to_theta))

        if len(img.shape) == 3:
            base_height, base_width, channel        = img.shape

        if len(img.shape) == 2:
            base_height, base_width = img.shape
            channel = 1

        # height = int(round(width * np.tan(fov_v / 2) / np.tan(fov_h / 2), 0))


        if len(img.shape) == 3:
            new_img = np.zeros((height, width, channel), np.uint8)
        if len(img.shape) == 2:
            new_img = np.zeros((height, width))

        fov_h = math.radians(fov_h_deg)
        fov_v = math.atan((height * math.tan((fov_h / 2)) / width)) * 2

        DI = np.ones((height * width, 3), np.int)

        # matrix to spherical coordinates
        trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                          [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])



        # DI: pixel row/column number
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        DI[:, 0] = xx.reshape(height * width)
        DI[:, 1] = yy.reshape(height * width)

        v = np.ones((height * width, 3), np.float)

        v[:, :2] = np.dot(DI, trans.T)  # to spherical coordinates
        v = np.dot(v, m.T)

        diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
        theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
        phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

        ey = np.rint(theta * base_height / np.pi).astype(np.int)
        ex = np.rint(phi * base_width / (2 * np.pi)).astype(np.int)

        ex[ex >= base_width] = base_width - 1
        ey[ey >= base_height] = base_height - 1

        new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]

        basename = f'{self.panoId}_{int(math.degrees(to_theta))}_{int(math.degrees(to_phi))}_{int(math.degrees(pitch))}.jpg'
        new_name = os.path.join(self.saved_path, basename)

        if len(img.shape) == 3:
            cv2.imwrite(new_name,  cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        if len(img.shape) == 2:
            cv2.imwrite(new_name.replace(".jpg", ".tif"), new_img)

        return new_img


    def save_image(self, np_img, new_name, worldfile):
        ext_name = "." + new_name[-3] + new_name[-1] + 'w'
        worldfile_name = new_name.replace(".tif", ext_name)
        saved_path = os.path.dirname(new_name)
        # save image
        if saved_path != "":
            try:
                if not os.path.exists(saved_path):
                    os.mkdir(saved_path)
                try:
                    im = Image.fromarray(np_img)
                except Exception as e:
                    # print(e)
                    im = Image.fromarray(np_img.astype(np.uint8))
                im.save(new_name)

                with open(worldfile_name, 'w') as wf:
                    for line in worldfile:
                        # print(line)
                        wf.write(str(line) + '\n')
            except Exception as e:
                print("Error in save_image(): ", self.saved_path, e)



    def clip_depthmap(self, to_theta=0, to_phi=0, width=1024, height=768, fov_h_deg=90, zoom=5, img_type="depthmap",
                  saved_path=os.getcwd()):
        if img_type == "pano":
            img = self.get_panorama(zoom=zoom)['image']
        if img_type == "depthmap":
            img = self.get_depthmap(zoom=zoom)['depthMap']
            pano_h, pano_w = self.jdata['Data']['level_sizes'][zoom][0]
            img = Image.fromarray(img).resize((pano_w, pano_h), Image.BILINEAR)
            img = np.array(img)

        self.saved_path = saved_path

        pitch = self.jdata['Projection']['tilt_pitch_deg']
        theta = self.jdata['Projection']['tilt_yaw_deg']
        pitch = math.radians(pitch)
        theta = math.radians(theta)
        to_theta = math.radians(to_theta)
        theta = (theta - pi / 2)

        # to_theta = math.radians(to_theta)
        to_phi = math.radians(to_phi)

        # rotation matrix
        m = np.eye(3)
        m = m.dot(utils.rotate_z(pitch))
        m = m.dot(utils.rotate_x(theta))
        m = m.dot(utils.rotate_y(to_phi))

        m = m.dot(utils.rotate_x(to_theta))

        if len(img.shape) == 3:
            base_height, base_width, channel = img.shape

        if len(img.shape) == 2:
            base_height, base_width = img.shape
            channel = 1

        # height = int(round(width * np.tan(fov_v / 2) / np.tan(fov_h / 2), 0))

        if len(img.shape) == 3:
            new_img = np.zeros((height, width, channel), np.uint8)
        if len(img.shape) == 2:
            new_img = np.zeros((height, width))

        fov_h = math.radians(fov_h_deg)
        fov_v = math.atan((height * math.tan((fov_h / 2)) / width)) * 2

        DI = np.ones((height * width, 3), np.int)
        trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                          [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        DI[:, 0] = xx.reshape(height * width)
        DI[:, 1] = yy.reshape(height * width)

        v = np.ones((height * width, 3), np.float)

        v[:, :2] = np.dot(DI, trans.T)
        v = np.dot(v, m.T)

        diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
        theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
        phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

        ey = np.rint(theta * base_height / np.pi).astype(np.int)
        ex = np.rint(phi * base_width / (2 * np.pi)).astype(np.int)

        ex[ex >= base_width] = base_width - 1
        ey[ey >= base_height] = base_height - 1

        new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]

        basename = f'{self.panoId}_{int(math.degrees(to_theta))}_{int(math.degrees(to_phi))}_{int(math.degrees(pitch))}.jpg'
        new_name = os.path.join(self.saved_path, basename)

        if len(img.shape) == 3:
            cv2.imwrite(new_name, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        if len(img.shape) == 2:
            cv2.imwrite(new_name.replace(".jpg", ".tif"), new_img)

        return new_img

    def get_image_from_headings(self, saved_path, phi_list=[], heading_list=[], prefix="", fov=30, height=768, width=768,
                                override=False):

        if len(phi_list) > 0:
            pano_yaw_deg = self.jdata['Projection']['pano_yaw_deg']
            heading_list = [(pano_yaw_deg + p) % 360 for p in phi_list]


        for h in heading_list:
            self.getImagefrmAngle(self.lon, self.lat, saved_path=saved_path,
                                   prefix=self.panoId, yaw=pano_yaw_deg + h, fov=fov, height=768, width=768, override=override)



