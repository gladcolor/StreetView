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

# Geospatial processing
from pyproj import Proj, transform
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

import numpy as np
from numpy import inf

# import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from skimage import io
import yaml

import PIL
from PIL import Image, features
from PIL import Image

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
        self.point_cloud = {"point_cloud": None, "zoom": None, "dm_mask": None}
        self.DEM = {"DEM": None, "zoom": None, "resolution": None}
        self.DOM = {"DOM": None, "zoom": None, "resolution": None}


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

                kernel = morphology.disk(2)
                dm_mask = morphology.erosion(depthMap['depthMap'], kernel)
                dm_mask = np.where(np.array(dm_mask) > 0, 1, 0).astype(int)
                # dm_mask = Image.fromarray(dm_mask)
                # dm_mask = dm_mask.resize((image_width, image_height), Image.NEAREST)
                # dm_mask = np.array(dm_mask)
                ground_mask = np.where(depthMap['normal_vector_map'][:, :, 2] < 10, 1, 0).astype(int)

                # interpolation using linear regression
                row_num =  self.depthmap['width']
                hc = depthMap['depthMap'][-1][int(self.depthmap['width']/2)]

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

                self.depthmap['depthMap'] = depthMap['depthMap']
                self.depthmap['dm_mask'] = dm_mask
                self.depthmap['ground_mask'] = ground_mask
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
                return self.depthmap
            except Exception as e:
                logger.exception("Error in get_depthmap(): %s", e)

        return self.depthmap

    # unfinished............
    def get_DEM(self, width = 40, height = 40, resolution=0.4, zoom=0):  # return: numpy array,

        if (self.DEM['DEM'] is None) or (self.DEM['zoom'] != zoom):
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None
                P = self.get_point_cloud(zoom=zoom, color=True)['point_cloud']

                # filter  points
                # keep the ground points.
                P = P[P[:, -1] < 10]
                # width2 = width *1.1
                # height2 = height * 1.1
                # r = math.sqrt(width**2 + height**2)
                # P = P[P[:, 3] < r]
                P = P[P[:, 0] < width/2]
                P = P[P[:, 0] > -width/2]
                P = P[P[:, 1] < height/2]
                P = P[P[:, 1] > -height/2]

                dem_coarse_resolution = resolution



                P_col = (P[:, 0]/dem_coarse_resolution + int(width / dem_coarse_resolution / 2)).astype(int)
                P_row = (int(height / dem_coarse_resolution / 2) - P[:, 1] / dem_coarse_resolution).astype(int)

                # dem_coarse_np = np.ones((int(height / dem_coarse_resolution), int(width / dem_coarse_resolution))) * -999
                dem_coarse_np = np.zeros((int(height / dem_coarse_resolution), int(width / dem_coarse_resolution)))
                # dem_coarse_np = np.zeros((int(height / dem_coarse_resolution), int(width / dem_coarse_resolution), 3))  # color image
                dem_coarse_np[P_row, P_col] = P[:, 2]
                # dem_coarse_np[P_row, P_col] = P[:, 4:7]    # color image

                grid_col = np.linspace(-width/2,  width/2,  dem_coarse_np.shape[1])
                grid_row = np.linspace(height/2, -height/2, dem_coarse_np.shape[0])



                resolution = 0.03
                w = int(width / resolution)
                h = int(height / resolution)
                dem_fined = np.array(Image.fromarray(dem_coarse_np).resize((w, h), Image.LINEAR))

                grid_col = np.linspace(-width/2,  width/2, w)
                grid_row = np.linspace(height/2, -height/2, h)

                xx, yy = np.meshgrid(grid_col, grid_row)


                kernel = morphology.disk(1)
                dem_mask = np.where(np.array(dem_coarse_np) < 0, 1, 0).astype(int)
                dem_mask = morphology.erosion(dem_mask, kernel)

                dem_mask = np.array(Image.fromarray(dem_mask).resize((w, h), Image.LINEAR))

                # dem_coarse_np = dem_fined

                # DEM_points = np.concatenate([xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1), dem_coarse_np.ravel().reshape(-1,1)], axis=1)
                # DEM_points = np.concatenate([DEM_points, np.zeros((len(DEM_points), 3))], axis=1)
                # need_colors = DEM_points
                # need_colors = DEM_points[DEM_points[:, 2] < 0]  # remove zero points.
                # interp = interpolate.interp2d(grid_col, grid_row, dem_coarse_np,
                #                               kind='linear')  # 'cubic' will distord the distance.

                # distance = interp(phi1, theta1)[0]


                # theta, phi = self.XYZ_to_spherical(need_colors)

                # find the pixels
                # colors = self.find_pixel_to_thetaphi(theta, phi)

                # DEM_points [DEM_points[:, 2] < 0, 3:6] = colors
                # DEM_points [:, 3:6] = colors

                # self.DEM["DEM"] = dem_coarse_np
                # DEM_points = DEM_points[DEM_points[:, 2] < 0] # remove zero points.


                # DEM_points[dem_mask.ravel() == 0, 3:6] = np.array([0, 0, 0])

                # self.DEM["DEM"] = DEM_points
                self.DEM["DEM"] = dem_coarse_np
                self.DEM["resolution"] = resolution
                self.DEM['zoom'] = int(zoom)

                # crs_local = CRS.from_proj4(f"+proj=tmerc +lat_0={self.lat} +lon_0={self.lon} +datum=WGS84 +units=m +no_defs")
                crs_local = CRS.from_proj4(f'+proj=tmerc +lat_0=38.83333333333334 +lon_0=-74.5 +k=0.9999 +x_0=150000 +y_0=0 +ellps=GRS80 +units=m +no_defs')
                print(crs_local)
                # transform = utils.epsg_transform(4326, 102005)  # USA_Contiguous_Equidistant_Conic, 102005
                # transform = utils.epsg_transform(4326, 103105)  # New Jersey state plane, meter. Error, no 103105.
                transformer = utils.epsg_transform(4326, crs_local)  # New Jersey state plane, meter

                x_m, y_m = transformer.transform(self.lat, self.lon)



                # if self.saved_path != "":
                #     if not os.path.exists(self.saved_path):
                #         os.mkdir(self.saved_path)
                #     # im = Image.fromarray(self.DEM['DEM'])
                #     im = Image.fromarray(self.DEM['DEM'].astype("uint8"), "RGB")
                #     new_name = os.path.join(self.saved_path, self.panoId + "_DEM+.tif")
                #     worldfile_name = new_name.replace(".tif", ".tfw")
                #     worldfile = [resolution, 0, 0, -resolution, x_m - width/2, y_m + height/2]
                #     im.show()
                #     im.save(new_name)
                #
                #     with open(worldfile_name, 'w') as wf:
                #         for line in worldfile:
                #             # print(line)
                #             wf.write(str(line) + '\n')

                # coarse_idx = np.argwhere(dem_coarse_np > -999)
                # coarse_values = dem_coarse_np[dem_coarse_np > -999]
                # dem_np = np.zeros((int(height / resolution), int(width / resolution)))

                # dm_mask = self.point_cloud["dm_mask"]

                # empty_idx = np.argwhere(dem_coarse_np == -999)
                # print("empty idx shape:", empty_idx.shape)

                # mask_values =

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

                # dem_np = np.zeros((int(height / resolution), int(width / resolution)))
                # dem_np[P_row, P_col] =  P[:,  2]

                # P = P[::20]


                # P = P[P[:, 3] < r]
                # P = P[P[:, 3] > 0]


                # print(P.shape)
                # print(coarse_values.shape)

                # DEM_values


                # self.DEM['DEM'] = np.concatenate([xxx.ravel().reshape(-1, 1), yyy.ravel().reshape(-1, 1), znew.ravel().reshape(-1, 1)], axis=1)
                # self.DEM['DEM'] = np.concatenate([coarse_idx[:,0].reshape(-1, 1) * dem_coarse_resolution, \
                #                                   coarse_idx[:, 1].reshape(-1, 1)* dem_coarse_resolution, \
                #                                   coarse_values.ravel().reshape(-1, 1)], axis=1)
                # self.DEM['DEM'] = P

            except Exception as e:
                logger.exception("Error in get_DEM(): %s", e)

        return self.DEM

    def XYZ_to_spherical(self, XYZs):
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
        Z = XYZs[:, 2]

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

    def find_pixel_to_thetaphi(self, theta, phi, zoom=4):
        ''':argument
        theata, phi: numpy array
        '''
        panorama = self.get_panorama(zoom=zoom)['image']
        image_height, image_width, channel = panorama.shape

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

    def get_DOM(self, width = 40, height = 40, resolution=0.03, zoom=4):  # return: numpy array,

        if (self.DOM['DOM'] is None) or (self.DEM['zoom'] != zoom):
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None
                dem_coarse_np = self.get_DEM(resolution=0.4, zoom=4)['DEM']


                # interpolation using Kriging
                cols = dem_coarse_np.shape[1]
                rows = dem_coarse_np.shape[0]
                idx = np.argwhere(dem_coarse_np)
                z = dem_coarse_np[dem_coarse_np < 0]
                xx, yy = idx[:,1].astype(float), idx[:,0].astype(float)
                OK = OrdinaryKriging(
                    xx,
                    yy,
                    z,
                    variogram_model="linear",
                    verbose=False,
                    enable_plotting=False,
                )
                z, ss = OK.execute("grid", np.arange(0.0, cols, 1.0), np.arange(0.0, rows, 1.0))

                # interpolation using linear regression



                dem_coarse_np = z


                w = int(width / resolution)
                h = int(height / resolution)



                dem_refined = Image.fromarray(dem_coarse_np).resize((int(width/resolution), int(height/resolution)), Image.LINEAR)
                dem_refined = np.array(dem_refined)

                grid_col = np.linspace(-width / 2, width / 2, w)
                grid_row = np.linspace(height / 2, -height / 2, h)
                xx, yy = np.meshgrid(grid_col, grid_row)

                kernel = morphology.disk(1)
                dem_mask = np.where(np.array(dem_coarse_np) < 0, 1, 0).astype(int)
                dem_mask = morphology.erosion(dem_mask, kernel)
                dem_mask = np.array(Image.fromarray(dem_mask).resize((w, h), Image.LINEAR))



                DEM_points = np.concatenate(
                    [xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1), dem_refined.ravel().reshape(-1, 1)],
                    axis=1)
                DEM_points = np.concatenate([DEM_points, np.zeros((len(DEM_points), 3))], axis=1) # add color columns
                # need_colors = DEM_points
                need_colors = DEM_points[DEM_points[:, 2] < 0]  # remove zero points.
                theta, phi = self.XYZ_to_spherical(need_colors)
                colors = self.find_pixel_to_thetaphi(theta, phi)
                DEM_points [DEM_points[:, 2] < 0, 3:6] = colors
                DEM_points[dem_mask.ravel() == 0, 3:6] = np.array([0, 0, 0])

                self.DOM['DOM'] = DEM_points

                crs_local = CRS.from_proj4(f'+proj=tmerc +lat_0=38.83333333333334 +lon_0=-74.5 +k=0.9999 +x_0=150000 +y_0=0 +ellps=GRS80 +units=m +no_defs')
                print(crs_local)
                # transform = utils.epsg_transform(4326, 102005)  # USA_Contiguous_Equidistant_Conic, 102005
                # transform = utils.epsg_transform(4326, 103105)  # New Jersey state plane, meter. Error, no 103105.
                transformer = utils.epsg_transform(4326, crs_local)  # New Jersey state plane, meter

                x_m, y_m = transformer.transform(self.lat, self.lon)

                im = DEM_points[:, 3:6].reshape((w, h, 3))

                if self.saved_path != "":
                    if not os.path.exists(self.saved_path):
                        os.mkdir(self.saved_path)
                    # im = Image.fromarray(self.DEM['DEM'])
                    im = Image.fromarray(im.astype("uint8"), "RGB")
                    new_name = os.path.join(self.saved_path, self.panoId + "_DOM.tif")
                    worldfile_name = new_name.replace(".tif", ".tfw")
                    worldfile = [resolution, 0, 0, -resolution, x_m - width/2, y_m + height/2]
                    im.show()
                    im.save(new_name)

                    with open(worldfile_name, 'w') as wf:
                        for line in worldfile:
                            # print(line)
                            wf.write(str(line) + '\n')

                # im = Image.fromarray(new_img)
                # im.show()

            except Exception as e:
                logger.exception("Error in get_DOM(): %s", e)

        return self.DOM

    def get_point_cloud(self, zoom=0, distance_threshole=40, color=True, saved_path=""):  # return: numpy array
        ''':param
        pano_zoom: -1: not colorize the point cloud
        '''

        if (self.point_cloud['point_cloud'] is None) or (self.point_cloud['zoom'] != zoom):
            try:
                if self.jdata is None:
                    logging.info("Jdata is None: %s.", self.panoId)
                    return None

                depthmap = self.get_depthmap()['depthMap']

                image_width = self.jdata['Data']['level_sizes'][zoom][0][1]
                image_height = self.jdata['Data']['level_sizes'][zoom][0][0]

                dm_resized = Image.fromarray(depthmap).resize((image_width, image_height), Image.LINEAR)
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

                normalvector = Image.fromarray(self.depthmap['plane_map']).resize((image_width, image_height), Image.NEAREST)
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

                # keep the ground points.
                # P = P[P[:, -1] < 10]

                # theta2, phi2 = self.XYZ_to_spherical(P)


                # keep the ground points.
                # P = P[P[:, -1] < 10]

                # P = P[:, :6]
                self.point_cloud['point_cloud'] = P
                self.point_cloud['zoom'] = int(zoom)

                return self.point_cloud

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
                    # old_img.close()

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

                # if int(zoom) == 0:
                #     target = target.crop((0, 0, image_width, image_height))
                target = target.crop((0, 0, image_width, image_height))


                if self.saved_path != "":
                    if not os.path.exists(self.saved_path):
                        os.mkdir(self.saved_path)
                    target.save(new_name)

                self.panorama["image"] = np.array(target)
                self.panorama['zoom'] = int(zoom)

            return self.panorama

        except Exception as e:
            logger.exception("Error in getPanoJPGfrmPanoId(): %s", e)
            return None




    def clip_pano(self, to_theta=0, to_phi=0, width=1024, height=768, fov_h_deg=90, zoom=5, type="pano", saved_path=os.getcwd()):
        if type == "pano":
            img = self.get_panorama(zoom=zoom)['image']
        if type == "depthmap":
            img = self.get_depthmap()['depthMap']

        self.saved_path = saved_path




        pitch = self.jdata['Projection']['tilt_pitch_deg']
        theta = self.jdata['Projection']['tilt_yaw_deg']
        pitch = math.radians(pitch)
        theta = math.radians(theta)
        to_theta = theta
        to_theta = (to_theta - pi / 2 )

        # to_theta = math.radians(to_theta)
        to_phi = math.radians(to_phi)

        # rotation matrix
        m = np.eye(3)
        m = m.dot(utils.rotate_z(pitch))
        m = m.dot(utils.rotate_x(to_theta))
        m = m.dot(utils.rotate_y(to_phi))

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

        cv2.imwrite(new_name,  cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

        return new_img

