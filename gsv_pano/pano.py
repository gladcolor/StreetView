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

# Geospatial processing
from pyproj import Proj, transform
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage import io
import yaml

import PIL
from PIL import Image, features
import cv2

from io import BytesIO
import random

import requests
import urllib.request
import urllib
import logging


import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sqlite3
from bs4 import BeautifulSoup
WINDOWS_SIZE = '100, 100'
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
Loading_time = 5

import utils

import logging.config


logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='Pano.log',
                filemode='w')



def setup_logging(default_path='log_config.yaml', logName='', default_level=logging.DEBUG):
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
logger = logging.getLogger('console_only')

class GSV_pano(object):
    def __init__(self, panoId="", request_lon=None, request_lat=None, request_address='', saved_path=''):
        self.panoId = panoId  # test case: BM1Qt23drK3-yMWxYfOfVg
        self.request_lon = request_lon  # test case: -74.18154077638651
        self.request_lat = request_lat  # test case: 40.73031168738437
        self.request_address = request_address # test case: '93-121 W Kinney St, Newark, NJ 07102'
        self.jdata = None  # json object
        self.lat = None
        self.lon = None
        self.saved_path = saved_path

        self.depthmap = {"depthMap":None, "width":None, "height": None, "plane_idx_map": None, "plane_map": None}
        self.panorama ={"image": None, "zoom": None}
        self.point_cloud = {"point_cloud": None, "zoom": None}
        self.DEM = {"DEM": None, "zoom": None}


        try:

            if request_lat and request_lon:
                if (-180 <= request_lon <= 180) and (-90 <= request_lat <= 90):
                    self.panoId, self.lon, self.lat = self.getPanoIDfrmLonlat(request_lon, request_lat)

            self.jdata = self.getJsonfrmPanoID(panoId=self.panoId, dm=1, saved_path=self.saved_path)
            self.lon = self.jdata['Location']['lng']
            self.lat = self.jdata['Location']['lat']

        except Exception as e:
            logging.exception("Error in GSV_pano _init__(): %s", e)


    def getPanoIDfrmLonlat(self, lon, lat):
        url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
        url = url.format(lat, lon)
        resp = requests.get(url, proxies=None)
        line = resp.text.replace("/**/_xdc_._v2mub5 && _xdc_._v2mub5( ", "")[:-2]

        if len(line) > 1000:
            try:
                jdata = json.loads(line)
                panoId = jdata[1][1][1]
                pano_lon = jdata[1][5][0][1][0][3]
                pano_lat = jdata[1][5][0][1][0][2]
                return panoId, pano_lon, pano_lat
            except:
                logging.exception("Error in getPanoIDfrmLonlat().", e)
                return 0, 0, 0

        else:  # if there is no panorama
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
        url = "https://www.google.com/maps/photometa/v1?authuser=0&hl=zh-CN&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1szh-CN!2sus!3m3!1m2!1e2!2s{}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3"
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


    def get_depthmap(self, saved_path=""):  # return: numpy array

        if self.depthmap['depthMap'] is None:
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None

                depthMapData = utils.parse(self.jdata['model']['depth_map'])
                # parse first bytes to describe data
                header = utils.parseHeader(depthMapData)
                # parse bytes into planes of float values
                self.depthmap['width'] = header['width']
                self.depthmap['height'] = header['height']

                data = utils.parsePlanes(header, depthMapData)
                # compute position and values of pixels
                depthMap = utils.computeDepthMap(header, data["indices"], data["planes"])
                self.depthmap['depthMap'] = depthMap['depthMap']
                self.depthmap['plane_map'] = depthMap['normal_vector_map']
                self.depthmap['plane_idx_map'] = depthMap['plane_idx_map']

                if len(saved_path) > 0:
                    if os.path.exists(saved_path):
                        os.path.mkdir()
                    new_name = os.path.join(saved_path, self.jdata['Location']['panoId'] + ".tif")
                    im = depthMap["depthMap"]

                    im[np.where(im == max(im))[0]] = 0

                    im = im.reshape((depthMap["height"], depthMap["width"]))  # .astype(int)
                    # display image
                    img = PIL.Image.fromarray(im)
                    img.save(new_name)
                return self.depthmap['depthMap']
            except Exception as e:
                logger.exception("Error in get_depthmap(): %s", e)

        return self.depthmap

    def get_DEM(self, saved_path=""):  # return: numpy array

        if self.depthmap['DEM'] is None:
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None

                depthmap = self.get_depthmap()

                if len(saved_path) > 0:
                    if os.path.exists(saved_path):
                        os.path.mkdir()
                    new_name = os.path.join(saved_path, self.jdata['Location']['panoId'] + ".tif")
                    im = depthMap["depthMap"]

                    im[np.where(im == max(im))[0]] = 0

                    im = im.reshape((depthMap["height"], depthMap["width"]))  # .astype(int)
                    # display image
                    img = PIL.Image.fromarray(im)
                    img.save(new_name)
                return self.depthmap['depthMap']
            except Exception as e:
                logger.exception("Error in get_depthmap(): %s", e)

        return self.depthmap['DEM']

    def get_point_cloud(self, w=20, h=20, zoom=0, color=True, saved_path=""):  # return: numpy array
        ''':param
        pano_zoom: -1: not colorize the point cloud
        '''

        if self.point_cloud['point_cloud'] is None:
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None
                r = math.sqrt(w**2 + h**2)
                depthmap = self.get_depthmap()['depthMap']


                # filter  points
                depthmap = np.where(depthmap < r, depthmap, 0)
                depthmap_idx = np.argwhere(0 < depthmap  )
                depthmap_value = depthmap[np.where(0 < depthmap)]

                image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
                image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

                dm_resized = Image.fromarray(depthmap).resize((image_width, image_height), Image.LINEAR)
                kernel = morphology.disk(2)
                dm_mask = morphology.erosion(depthmap, kernel)
                dm_mask = np.where(np.array(dm_mask) > 0, 1, 0).astype(int)
                dm_mask = Image.fromarray(dm_mask)
                dm_mask = dm_mask.resize((image_width, image_height), Image.NEAREST)
                dm_mask = np.array(dm_mask)
                dm_mask[0, :] = 1  # show the camera center.

                tilt_yaw_deg = self.jdata['Projection']['tilt_yaw_deg']
                pano_yaw_deg = self.jdata['Projection']['pano_yaw_deg']
                tilt_pitch_deg = self.jdata['Projection']['tilt_pitch_deg']
                elevation_egm96_m = self.jdata['Location']['elevation_egm96_m']

                dm_np = np.array(dm_resized).astype(float)
                dm_np = dm_np * dm_mask

                img = self.get_panorama(zoom=zoom)

                img_np = np.array(img)[0:image_height, :, :]
                colors = img_np.reshape(-1, 3)

                normalvector = Image.fromarray(self.depthmap['plane_map']).resize((image_width, image_height), Image.NEAREST)
                normalvector_np = np.array(normalvector)
                normalvectors = normalvector_np.reshape(-1, 3)

                plane_idx = Image.fromarray(self.depthmap['plane_idx_map']).resize((image_width, image_height), Image.NEAREST)
                plane_np = np.array(plane_idx)
                planes = plane_np.reshape(-1, 1)

                # print(colors.shape)
                # print(colors)
                # print(img_np.shape)
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
                x = R * phis_cos
                y = R * phis_sin

                P = np.concatenate([y.ravel().reshape(-1, 1), x.ravel().reshape(-1, 1), z.ravel().reshape(-1, 1)],
                                   axis=1)

                # rotate_x_radian = (90 - tilt_yaw_deg) / 180 * math.pi
                # rotate_z_radian = (90 - pano_yaw_deg) / 180 * math.pi
                # rotate_y_radian = (-tilt_pitch_deg) / 180 * math.pi  # should  be negative according to the observation of highway ramp.
                #
                rotate_x_radian = math.radians(90 - tilt_yaw_deg)
                rotate_y_radian = math.radians(
                    -tilt_pitch_deg)  # should  be negative according to the observation of highway ramp.
                rotate_z_radian = math.radians(90 - pano_yaw_deg)

                P = P.dot(utils.rotate_x(rotate_x_radian))  # math.radians(90 - tilt_yaw_deg)  # roll
                P = P.dot(utils.rotate_y(rotate_y_radian))  # math.radians(-tilt_pitch_deg)  # pitch
                P = P.dot(utils.rotate_z(rotate_z_radian))  # math.radians(90 - pano_yaw_deg)  # yaw

                if color:
                    P = np.concatenate([P, colors, planes, normalvectors, normalvectors[:, 2:3]], axis=1)

                P = np.concatenate([P, dm_np.ravel().reshape(-1, 1)], axis=1)
                P = P[np.where(dm_mask.ravel())]
                distance_threshole = 20
                P = P[P[:, -1] < distance_threshole]

                # keep the ground points.
                P = P[P[:, -2] < 10]

                # P = P[P[:, 2] < 8]

                # P = P[P[:, -1] < 30]

                # print(P.shape, P)

                P = P[:, :6]
                self.point_cloud['point_cloud'] = P
                self.point_cloud['zoom'] = int(zoom)

                # interpolate_fun = interpolate.interp2d(depthmap_idx[:, 0], depthmap_idx[:, 1], depthmap_value, kind='linear')

                # if len(saved_path) > 0:
                #     if os.path.exists(saved_path):
                #         os.path.mkdir()
                #     new_name = os.path.join(saved_path, self.jdata['Location']['panoId'] + ".tif")
                #     im = depthMap["depthMap"]
                #
                #     im[np.where(im == max(im))[0]] = 0
                #
                #     im = im.reshape((depthMap["height"], depthMap["width"]))  # .astype(int)
                #     # display image
                #     img = PIL.Image.fromarray(im)
                #     img.save(new_name)
                return self.point_cloud['point_cloud']
            except Exception as e:
                logger.exception("Error in get_depthmap(): %s", e)




    def get_panorama(self, prefix="", suffix="", zoom: int = 5) -> str:
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

            if (self.panorama['image'] is None) or (zoom != self.panorama['zoom']):
                # print(adcode)  # Works well.
                tile_width = self.jdata['Data']['tile_width']
                tile_height = self.jdata['Data']['tile_height']

                zoom = int(zoom)
                image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
                image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

                # passed
                column_cnt = np.ceil(image_width / tile_width).astype(int)
                row_cnt = np.ceil(image_height / tile_height).astype(int)

                new_name = os.path.join(self.saved_path, (prefix + self.panoId + suffix + '.jpg'))
                target = Image.new('RGB', (tile_width * column_cnt, tile_height * row_cnt))  # new graph

                if os.path.exists(new_name):
                    old_img = Image.open(new_name)
                    if old_img.size == (image_width, image_height):  # no need to download new image
                        column_cnt = 0
                        row_cnt = 0
                        target = old_img
                    old_img.close()

                for x in range(column_cnt):  # col
                    for y in range(row_cnt):  # row
                        num = random.randint(0, 3)
                        zoom = str(zoom)
                        url = 'https://geo' + str(
                            num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + self.panoId + '&output=tile&x=' + str(
                            x) + '&y=' + str(y) + '&zoom=' + zoom + '&nbt&fover=2'
                        file = urllib.request.urlopen(url)
                        image = Image.open(file)
                        image = image.resize((tile_width, tile_height))
                        target.paste(image, (tile_width * x, tile_height * y, tile_width * (x + 1), tile_height * (y + 1)))

                if prefix != "":
                    prefix += '_'
                if suffix != "":
                    suffix = '_' + suffix

                if int(zoom) == 0:
                    target = target.crop((0, 0, image_width, image_height))

                if self.saved_path != "":
                    if not os.path.exists(self.saved_path):
                        os.mkdir(self.saved_path)
                    target.save(new_name)

                self.panorama["image"] = target
                self.panorama['zoom'] = int(zoom)

            return self.panorama["image"]

        except Exception as e:
            logger.exception("Error in getPanoJPGfrmPanoId(): %s", e)
            return None

