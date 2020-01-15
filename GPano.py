"""
Designed by Huan Ning, gladcolor@gmail.com, 2019.09.04

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
import selenium
import os
import time
from io import BytesIO
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import requests
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
#
WINDOWS_SIZE = '100, 100'
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
Loading_time = 5
import sqlite3

web_driver_path = r'K:\Research\StreetView\Google_street_view\chromedriver.exe'
driver = webdriver.Chrome(executable_path=web_driver_path, chrome_options=chrome_options)
Process_cnt = 10

"""
Read Me 
lon/lat = longitude/latitude
The current objective of GPano class is to download the panorama image from Google Street View according to lon/lat.
Main workflow: get panorama image ID based on lon/lat -> according to the panorama id, get tiles of a Panorama image and then mosaic them 
Please implement all the methods. I have written some tips (not code) in the method body to assist you. -- Huan
"""


class GPano():
    # Obtain a panomaro image from Google Street View Map
    def getPanoZoom0frmID(self, panoId, saved_path):
        url = r'http://maps.google.com/cbk?output=tile&zoom=0&x=0&y=0&panoid=' + panoId
        # print(url)
        file = urllib.request.urlopen(url)
        image = Image.open(file)
        image.save(os.path.join(saved_path, panoId + '.jpg'))

    def getGSV_url_frm_lonlat(self, lon, lat, heading=0, tilt=90, fov=90):
        url_part1 = r'!3m6!1e1!3m4!1s'
        url_part2 = r'!2e0!7i16384!8i8192'
        heading = str(heading)
        url = f"https://www.google.com/maps/@{round(lat, 7)},{round(lon, 7)},3a,{fov}y,{heading}h,{tilt}t/data={url_part1}{url_part2}"
        return url

    def getGSV_url_frm_panoId(self, panoId, heading=0, tilt=90, fov=90):
        print(panoId)
        jdata = self.getJsonfrmPanoID(panoId)
        print(jdata)
        lon = jdata["Location"]["original_lng"]
        lat = jdata["Location"]["original_lat"]
        url_part1 = r'!3m6!1e1!3m4!1s'
        url_part2 = r'!2e0!7i16384!8i8192'
        heading = str(heading)
        url = f"https://www.google.com/maps/@{lat},{lon},3a,{fov}y,{heading}h,{tilt}t/data={url_part1}{url_part2}"
        return url

    def getDegreeOfTwoLonlat(self, latA, lonA, latB, lonB):

        """
        Args:
            point p1(latA, lonA)
            point p2(latB, lonB)
        Returns:
            bearing between the two GPS points,
            default: the basis of heading direction is north
            https://blog.csdn.net/zhuqiuhui/article/details/53180395

            This article shows the similar method
            Bearing from point B to A,

            https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
        """
        radLatA = math.radians(float(latA))
        radLonA = math.radians(float(lonA))
        radLatB = math.radians(float(latB))
        radLonB = math.radians(float(lonB))
        dLon = radLonB - radLonA
        y = math.sin(dLon) * cos(radLatB)
        x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
        brng = degrees(atan2(y, x))
        brng = (brng + 360) % 360
        return brng

    def getPanoJPGsfrmLonlats(self, list_lonlat, saved_path, prefix="", suffix="", zoom=4):
        """ Obtain panomara images from a list of lat/lon: [(lon, lat), ...]

        """
        statuses = []  # succeeded: 1; failed: 0
        # print(list_lonlat.pop(0))
        start_time = time.time()
        Cnt = 0
        Cnt_interval = 100

        while len(list_lonlat) > 0:
            lon, lat, row = list_lonlat.pop(0)
            print(lon, lat, row)
            prefix = row
            self.getPanoJPGfrmLonlat(lon, lat, saved_path, prefix, suffix)

            # start_time = time.time()
            # Cnt = 0
            # Cnt_interval = 100
            Cnt += 1

            if Cnt % Cnt_interval == (Cnt_interval - 1):
                print("Process speed: {} points / hour.".format(int(Cnt / (time.time() - start_time + 0.001) * 3600)))
        return statuses

    def readCoords_csv(self, file_path):
        df = pd.read_csv(file_path)
        return list(df.itertuples(index=False, name=None))
        # with open(file_path, 'r') as f:
        #     lines = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        #     # This skips the first row of the CSV file.
        #     # csvreader.next() also works in Python 2.
        #     # drop header
        #     next(lines)
        #     return [tuple(line) for line in lines]
        #     #for line in lines:
        #         #print(line)

    def formPolygon(self, coords):
        return Polygon(coords)

    def point_in_polygon(self, point, polygon):
        try:
            return point.within(polygon)
        except Exception as e:
            print("Error in point_in_polygon():", e)

    def getPanoJPGfrmLonlat0(self, lon: float, lat: float, saved_path: str, prefix="", suffix="",
                             zoom: int = 4) -> bool:
        """Reference:
            https://developers.google.com/maps/documentation/javascript/streetview
            See the part from "Providing Custom Street View Panoramas" section.
            Get those tiles and mosaic them to a large image.
            The url of a tile:
            https://geo2.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&panoid=CJ31ttcx7ez9qcWzoygVqA&output=tile&x=1&y=1&zoom=4&nbt&fover=2
            Make sure randomly use geo0 - geo3 server.
            When zoom=4, a panorama image have 6 rows, 13 cols.
        """
        status = 0
        # PanoID, lon_pano, lat_pano = self.getPanoIDfrmLonlat(lat, lon, zoom=4)

        adcode = self.getPanoIDfrmLonlat(lon, lat)

        if str(adcode[0]) == str(0):
            print(adcode[0], "is not a PanoID.")
            return status

        print(adcode)  # Works well.
        for x in range(30):  # test for the size of map, column, x
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[
                          0] + '&output=tile&x=' + str(
                    x) + '&y=' + str(0) + '&zoom=4&nbt&fover=2'
                file = urllib.request.urlopen(url)
                image = Image.open(file)
            except OSError:
                m = x
                break

        for x in range(30):  # test for the size of map, row, y
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[
                          0] + '&output=tile&x=' + str(
                    0) + '&y=' + str(x) + '&zoom=4&nbt&fover=2'
                file = urllib.request.urlopen(url)
                image = Image.open(file)
            except OSError:
                n = x
                break
        print('col, row:', m, n)

        UNIT_SIZE = 512
        try:
            target = Image.new('RGB', (UNIT_SIZE * m, UNIT_SIZE * n))  # new graph
            for x in range(m):  # list
                for y in range(n):  # row
                    num = random.randint(0, 3)
                    url = 'https://geo' + str(
                        num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[
                              0] + '&output=tile&x=' + str(
                        x) + '&y=' + str(y) + '&zoom=4&nbt&fover=2'
                    file = urllib.request.urlopen(url)
                    image = Image.open(file)
                    target.paste(image, (UNIT_SIZE * x, UNIT_SIZE * y, UNIT_SIZE * (x + 1), UNIT_SIZE * (y + 1)))
            if prefix != "":
                prefix += '_'
            if suffix != "":
                suffix = '_' + suffix
            mapname = os.path.join(saved_path,
                                   (prefix + adcode[1] + '_' + adcode[2] + '_' + adcode[0] + suffix + '.jpg'))
            target.save(mapname)
            status = 1
        except Exception as e:
            print("Error in getPanoJPGfrmLonlat():", e)
            status = 0

        return status

    def getPanoJPGfrmLonlat(self, lon: float, lat: float, saved_path: str, prefix="", suffix="", zoom: int = 4) -> bool:
        """Reference:
            https://developers.google.com/maps/documentation/javascript/streetview
            See the part from "Providing Custom Street View Panoramas" section.
            Get those tiles and mosaic them to a large image.
            The url of a tile:
            https://geo2.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&panoid=CJ31ttcx7ez9qcWzoygVqA&output=tile&x=1&y=1&zoom=4&nbt&fover=2
            Make sure randomly use geo0 - geo3 server.
            When zoom=4, a panorama image have 6 rows, 13 cols.
        """
        status = 0
        # PanoID, lon_pano, lat_pano = self.getPanoIDfrmLonlat(lat, lon, zoom=4)

        adcode = self.getPanoIDfrmLonlat(lon, lat)

        if str(adcode[0]) == str(0):
            print(adcode[0], "is not a PanoID.")
            return status

        print(adcode)  # Works well.
        for x in range(36):  # test for the size of map, column, x
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[
                          0] + '&output=tile&x=' + str(
                    x) + '&y=' + str(0) + '&zoom=' + zoom + '&nbt&fover=2'
                file = urllib.request.urlopen(url)
                image = Image.open(file)
            except OSError:
                m = x
                break

        for x in range(36):  # test for the size of map, row, y
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[
                          0] + '&output=tile&x=' + str(
                    0) + '&y=' + str(x) + '&zoom=4&nbt&fover=2'
                # response = requests.get(url)
                # image = Image.open(BytesIO(response.content))
                file = urllib.request.urlopen(url)
                image = Image.open(file)
            except OSError:
                n = x
                break
        print('col, row:', m, n)

        UNIT_SIZE = 512
        try:
            target = Image.new('RGB', (UNIT_SIZE * m, UNIT_SIZE * n))  # new graph
            for x in range(m):  # list
                for y in range(n):  # row
                    num = random.randint(0, 3)
                    url = 'https://geo' + str(
                        num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[
                              0] + '&output=tile&x=' + str(
                        x) + '&y=' + str(y) + '&zoom=4&nbt&fover=2'
                    file = urllib.request.urlopen(url)
                    image = Image.open(file)
                    target.paste(image, (UNIT_SIZE * x, UNIT_SIZE * y, UNIT_SIZE * (x + 1), UNIT_SIZE * (y + 1)))
            if prefix != "":
                prefix += '_'
            if suffix != "":
                suffix = '_' + suffix
            mapname = os.path.join(saved_path,
                                   (prefix + adcode[1] + '_' + adcode[2] + '_' + adcode[0] + suffix + '.jpg'))
            target.save(mapname)
            status = 1
        except Exception as e:
            print("Error in getPanoJPGfrmLonlat():", e)
            status = 0

        return status

    def getPanoJPGfrmPanoId(self, panoId, saved_path: str, prefix="", suffix="", zoom: int = 4) -> bool:
        """Reference:
            https://developers.google.com/maps/documentation/javascript/streetview
            See the part from "Providing Custom Street View Panoramas" section.
            Get those tiles and mosaic them to a large image.
            The url of a tile:
            https://geo2.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&panoid=CJ31ttcx7ez9qcWzoygVqA&output=tile&x=1&y=1&zoom=4&nbt&fover=2
            Make sure randomly use geo0 - geo3 server.
            When zoom=4, a panorama image have 6 rows, 13 cols.
        """
        status = 0
        zoom = str(zoom)
        # PanoID, lon_pano, lat_pano = self.getPanoIDfrmLonlat(lat, lon, zoom=4)

        adcode = panoId

        images = []

        if str(adcode) == str(0):
            print(adcode, "is not a PanoID.")
            return status

        # print(adcode)  # Works well.
        for x in range(36):  # test for the size of map, column, x
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode + '&output=tile&x=' + str(
                    x) + '&y=' + str(0) + '&zoom=' + zoom + '&nbt&fover=2'
                # print(url)
                file = urllib.request.urlopen(url)
                image = Image.open(file)
                # images.append(image)

            except OSError:
                m = x
                break

        for x in range(36):  # test for the size of map, row, y
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode + '&output=tile&x=' + str(
                    0) + '&y=' + str(x) + '&zoom=' + zoom + '&nbt&fover=2'
                file = urllib.request.urlopen(url)
                image = Image.open(file)
            except OSError:
                n = x
                break

        url = 'https://geo' + str(
            num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode + '&output=tile&x=' + str(
            0) + '&y=' + str(0) + '&zoom=' + zoom + '&nbt&fover=2'
        file = urllib.request.urlopen(url)
        image = Image.open(file)

        UNIT_SIZE = 512

        try:
            target = Image.new('RGB', (UNIT_SIZE * m, UNIT_SIZE * n))  # new graph

            for x in range(m):  # col
                for y in range(n):  # row
                    num = random.randint(0, 3)
                    url = 'https://geo' + str(
                        num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode + '&output=tile&x=' + str(
                        x) + '&y=' + str(y) + '&zoom=' + zoom + '&nbt&fover=2'
                    file = urllib.request.urlopen(url)
                    image = Image.open(file)

                    # print('UNIT_SIZE:', UNIT_SIZE)
                    image = image.resize((UNIT_SIZE, UNIT_SIZE))
                    # print('col, row:', m, n)
                    # print('UNIT_SIZE * x, UNIT_SIZE * y, UNIT_SIZE * (x + 1), UNIT_SIZE * (y + 1):', UNIT_SIZE * x, UNIT_SIZE * y, )
                    target.paste(image, (UNIT_SIZE * x, UNIT_SIZE * y, UNIT_SIZE * (x + 1), UNIT_SIZE * (y + 1)))

            if prefix != "":
                prefix += '_'
            if suffix != "":
                suffix = '_' + suffix
            mapname = os.path.join(saved_path, (prefix + adcode + suffix + '.jpg'))
            target.save(mapname)
            status = 1
        except Exception as e:
            print("Error in getPanoJPGfrmPanoId():", e)
            status = 0

        return status

    def getPanosfrmLonlats_mp(self, list_lonlat_mp, saved_path, prefix="", suffix="", zoom=4, Process_cnt=4):
        """ Multi_processing version of getPanosfrmLonlats()
            Obtain panomara images from a list of lat/lon: [(lon, lat), ...]

        """
        statuses = []  # succeeded: 1; failed: 0
        pool = mp.Pool(processes=Process_cnt)
        for i in range(Process_cnt):
            pool.apply_async(self.getPanoJPGsfrmLonlats, args=(list_lonlat_mp, saved_path))
        pool.close()
        pool.join()

    # Obtain a panomara_ID according to lon/lat.
    def getPanoIDfrmLonlat0(self, lon: float, lat: float, ) -> (str, float, float):
        """ Obtain panomara_id from lat/lon.
            Use selenium to obtain the new url, which contains the panomara_id
            Initial url: https://www.google.com/maps/@39.9533555,-75.1544777,3a,90y,180h,90t/data=!3m6!1e1!3m4!1s!2e0!7i16384!8i8192
            New url returned by Google: https://www.google.com/maps/@39.9533227,-75.1544758,3a,90y,180h,90t/data=!3m6!1e1!3m4!1sAF1QipNWKtSlDw5M8fsZxdQnXtSw3zWOgMIY8fN_eEbv!2e10!7i5504!8i2752
            PanoID: AF1QipNWKtSlDw5M8fsZxdQnXtSw3zWOgMIY8fN_eEbv
            Function return the panomara_id and its lon/lon.
        """
        heading = 180
        tilt = 90
        fov = 90
        url_part1 = r'!3m6!1e1!3m4!1s'
        url_part2 = r'!2e0!7i16384!8i8192'
        url = f"https://www.google.com/maps/@{round(lat, 7)},{round(lon, 7)},3a,{fov}y,{heading}h,{tilt}t/data={url_part1}{url_part2}"

        # print(url)
        try:
            driver.get(url)
            time.sleep(4)
            new_url = driver.current_url
            end_new_url = new_url[new_url.find("data="):]
            # print(end_new_url)
            if len(end_new_url) < len(f'data={url_part1}{url_part2}'):  # fail to get the PanoID
                time.sleep(2)
            if len(end_new_url) < len(f'data={url_part1}{url_part2}'):  # fail to get the PanoID
                PanoID = 0
                lon_pano = 0
                lat_pano = 0
                print("No new url for: ", url)

            else:
                lat_pano, lon_pano = new_url.split(',')[:2]
                lat_pano = lat_pano.split("@")[1]
                pos1 = new_url.find(url_part1)
                # pos2 = new_url.find(url_part2)
                pos2 = new_url.find(url_part2[:5])
                PanoID = new_url[(pos1 + len(url_part1)):pos2]
                # print(url)
                print("getPanoIDfrmLonlat0() obtained new_url: ", new_url)

            return PanoID, lon_pano, lat_pano  # if cannot find the panomara, return (0, 0, 0)
        except Exception as e:
            print("Error in getPanoIDfrmLonlat()0", e)

    # Finished!
    def getPanoIDfrmLonlat(self, lon, lat):
        url = f'http://maps.google.com/cbk?output=json&ll={lat},{lon}'
        # print(url)
        r = requests.get(url)
        data = self.getPanoJsonfrmLonat(lon, lat)
        #
        # if not data:
        #     panoId = self.getPanoIDfrmLonlat0(lon, lat)
        #     print("data from getPanoIDfrmLonlat0(): ", panoId)

        if data == 0:
            return 0, 0, 0
        if 'Location' in data:
            return (data['Location']['panoId'], data['Location']['original_lng'], data['Location']['original_lat'])
        else:
            return 0, 0, 0

    def getPanoJsonfrmLonat(self, lon, lat):
        try:
            server_num = random.randint(0, 3)
            # url = f'http://maps.google.com/cbk?output=json&ll={lat},{lon}'

            url = f'https://geo{server_num}.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&output=json&ll={lat}%2C{lon}&dm=0'
            # print(url)
            r = requests.get(url)
            if not r.json():
                panoId, lon_p, lat_p = self.getPanoIDfrmLonlat0(lon, lat)
                url = f'http://maps.google.com/cbk?output=json&panoid={panoId}'
                # print(url)
                r = requests.get(url)

            return r.json()

        except Exception as e:
            print("Error in getPanoJsonfrmLonnat():", e)
            return 0

    def getImagefrmAngle(self, lon: float, lat: float, saved_path='', prefix='', suffix='', width=1024, height=768,
                         pitch=0, yaw=0):
        # w maximum: 1024
        # h maximum: 768
        server_num = random.randint(0, 3)
        lon = round(lon, 7)
        lat = round(lat, 7)
        height = int(height)
        pitch = int(pitch)
        width = int(width)

        if yaw > 360:
            yaw = yaw - 360
        if yaw < 0:
            yaw = yaw + 360

        url1 = f"https://geo{server_num}.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&output=thumbnail&thumb=2&w={width}" \
               f"&h={height}&pitch={pitch}&ll={lat}%2C{lon}&yaw={yaw}&thumbfov=90"

        suffix = str(suffix)
        prefix = str(prefix)
        if prefix != "":
            # print('prefix:', prefix)
            prefix = prefix + '_'
        if suffix != "":
            suffix = '_' + suffix

        try:
            file = urllib.request.urlopen(url1)
            image = Image.open(file)

            jpg_name = os.path.join(saved_path, (prefix + str(lon) + '_' + str(lat) + '_' + str(pitch) + '_' +
                                                 str('{:.2f}'.format(yaw)) + suffix + '.jpg'))
            if image.getbbox():
                if saved_path != '':
                    image.save(jpg_name)
                else:
                    # print(url1)
                    pass
                return image, jpg_name

        except Exception as e:
            print("Error in getImagefrmAngle() getting url1", e)
            print(url1)
            return 0, jpg_name

    def getImageclipfrmPano(self, panoId, saved_path='', prefix='', suffix='', width=1024, height=768, pitch=0, yaw=0):
        # w maximum: 1024
        # h maximum: 768
        server_num = random.randint(0, 3)
        jdata = self.getJsonfrmPanoID(panoId)
        lon = jdata["Location"]["original_lng"]
        lat = jdata["Location"]["original_lat"]
        height = int(height)
        pitch = int(pitch)
        width = int(width)

        if yaw > 360:
            yaw = yaw - 360
        if yaw < 0:
            yaw = yaw + 360

        url1 = f"https://geo{server_num}.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&output=thumbnail&thumb=2&w={width}" \
               f"&h={height}&pitch={pitch}&panoid={panoId}&yaw={yaw}&thumbfov=90"

        suffix = str(suffix)
        prefix = str(prefix)
        if prefix != "":
            # print('prefix:', prefix)
            prefix = prefix + '_'
        if suffix != "":
            suffix = '_' + suffix

        try:
            file = urllib.request.urlopen(url1)
            image = Image.open(file)

            jpg_name = os.path.join(saved_path, (prefix + str(lon) + '_' + str(lat) + '_' + str(pitch) + '_' +
                                                 str(int(yaw)) + suffix + '.jpg'))
            if image.getbbox():
                if saved_path != '':
                    image.save(jpg_name)
                    return image, jpg_name
                else:
                    return image
            # print(url2)
        except Exception as e:
            print("Error in getImagefrmAngle() getting url1", e)
            print(url1)
            return 0

    def getImagesfrmAngles(self, lonlat_list, saved_path='Panos', prefix='', suffix='', width=1024, height=768, pitch=0,
                           yaw=0):
        # w maximum: 1024
        # h maximum: 768
        start_time = time.time()
        Cnt = 0
        Cnt_interval = 100
        while len(lonlat_list) > 0:
            lon, lat, prefix, yaw = lonlat_list.pop(0)
            try:
                self.getImagefrmAngle(lon, lat, saved_path=saved_path, prefix=prefix, yaw=yaw)
                Cnt += 1
                if Cnt % Cnt_interval == (Cnt_interval - 1):
                    print(
                        "Process speed: {} points / hour.".format(int(Cnt / (time.time() - start_time + 0.001) * 3600)))

            except Exception as e:
                print("Error in getImagesfrmAngles():", e)

    def getImagesfrmAngles_mp(self, list_lonlat_mp, saved_path='Panos', Process_cnt=4, prefix='', suffix='', width=1024,
                              height=768, pitch=0, yaw=0):
        pool = mp.Pool(processes=Process_cnt)
        for i in range(Process_cnt):
            pool.apply_async(self.getPanoJPGsfrmLonlats, args=(list_lonlat_mp, saved_path))
        pool.close()
        pool.join()

    def readRoadSeedsPts_csv(self, file_path):
        df = pd.read_csv(file_path)
        return list(df.itertuples(index=False, name=False))

    def getNextJson(self, jdata, pre_panoId=""):
        try:
            yaw = float(jdata['Projection']['pano_yaw_deg'])
            links = jdata["Links"]
            # print("jdata[Links]: ", jdata["Links"])
            yaw_in_links = [float(link['yawDeg']) for link in links]
            # yaw_in_links = [float(link['yawDeg']) for link in links]
            # print("yaw_in_links: ", yaw_in_links)

            diff = [abs(yawDeg - yaw) for yawDeg in yaw_in_links]
            idx = diff.index(min(diff))
            # print(idx, diff)
            panoId = links[idx]['panoId']  # Warning: Not necessarily the second link node is the next panorama.
            # print("getNextJson: ", panoId, pre_panoId)
            if (panoId == pre_panoId) and (len(links) > 1):
                diff.pop(idx)
                links.pop(idx)
                idx = diff.index(min(diff))

                panoId = links[idx]['panoId']
                return self.getJsonfrmPanoID(panoId, dm=0)

            if len(links) == 1:
                return 0
            else:
                return self.getJsonfrmPanoID(panoId, dm=0)

        except Exception as e:
            print("Error in getNextJson(): ", e)
            print("panoId: ", pre_panoId)
            return 0

    def fileExists(self, path_list, basename):

        if not isinstance(path_list, list):
            path_list = [path_list]

        for path in path_list:
            if os.path.exists(os.path.join(path, basename)):
                return True
        return False

    def getLastJson(self, jdata, pre_panoId=""):
        try:
            yaw = float(jdata['Projection']['pano_yaw_deg'])
            links = jdata["Links"]
            # print("jdata[Links]: ", jdata["Links"])
            yaw_in_links = [float(link['yawDeg']) for link in links]
            # yaw_in_links = [float(link['yawDeg']) for link in links]
            # print("yaw_in_links: ", yaw_in_links)

            diff = [abs(yawDeg - yaw) for yawDeg in yaw_in_links]
            idx = diff.index(max(diff))
            # print(idx, diff)
            panoId = links[idx]['panoId']  # Warning: Not necessarily the second link node is the next panorama.

            if (panoId == pre_panoId) and (len(links) > 1):
                # print("getNextJson: ", panoId, pre_panoId)
                # print("idx: ", idx)
                diff.pop(idx)
                links.pop(idx)
                idx = diff.index(max(diff))
                # print("idx: ", idx)
                panoId = links[idx]['panoId']
                # print("links: ", links)
                # print("getNextJson: ", panoId, pre_panoId)
                return self.getJsonfrmPanoID(panoId, dm=0)

            if len(links) == 1:
                return 0
            else:
                return self.getJsonfrmPanoID(panoId, dm=0)

        except Exception as e:
            print("Error in getLastJson(): ", e)
            print(links)
            return "Error"

    def getImageclipsfrmJson(self, jdata, yaw_list=0, pitch_list=0, saved_path=''):
        try:
            panoId = jdata["Location"]["panoId"]
            heading = float(jdata["Projection"]["pano_yaw_deg"])
            lon = float(jdata["Location"]["original_lng"])
            lat = float(jdata["Location"]["original_lat"])

            if not isinstance(yaw_list, list):
                yaw_list = [yaw_list]

            if not isinstance(pitch_list, list):
                pitch_list = [pitch_list]

            for yaw in yaw_list:
                if yaw.isnumeric():
                    for pitch in pitch_list:
                        self.getImagefrmAngle(lon=lon, lat=lat, saved_path=saved_path, prefix=panoId, pitch=pitch,
                                              yaw=yaw + heading)

        except Exception as e:
            print("Error in getImageclipsfrmJson():", e)

    def go_along_road_forward(self, lon, lat, saved_path, yaw_list=0, pitch_list=0, steps=99999, polygon=None, zoom=5):
        lon = float(lon)
        lat = float(lat)
        if not isinstance(yaw_list, list):
            yaw_list = [yaw_list]

        if not isinstance(pitch_list, list):
            pitch_list = [pitch_list]

        if not self.point_in_polygon(Point((lon, lat)), polygon):
            print("Starting point is not in the polygon.")
            return

        step_cnt = 0
        next_pt = (lon, lat)
        lonlats = []
        next_panoId = self.getPanoIDfrmLonlat(lon, lat)[0]

        if next_panoId == 0:
            return lonlats

        pre_panoId = ""

        pt_lon = lon
        pt_lat = lat

        start_time = time.time()
        Cnt = 0
        Cnt_interval = 10

        try:
            while step_cnt < steps and self.point_in_polygon(Point(pt_lon, pt_lat), polygon):

                try:
                    try:

                        jdata = self.getJsonfrmPanoID(next_panoId, dm=1)
                    except Exception as e:
                        print("Error in getJsonfrmPanoID in  go_along_road_forward():", e)
                        print("Waiting for 5 seconds...")
                        time.sleep(5)
                        next_Json = self.getNextJson(jdata, pre_panoId)
                        if next_Json == 0:
                            return lonlats

                        next_lon = next_Json["Location"]["original_lng"]
                        next_lat = next_Json["Location"]["original_lat"]
                        next_panoId = next_Json["Location"]["panoId"]
                        pre_panoId = panoId

                        step_cnt += 1
                        next_pt = (next_lon, next_lat)
                        if len(pre_panoId) < 10:
                            return lonlats
                        if panoId == 0:
                            return lonlats

                        print('step_cnt: ', step_cnt)
                        continue

                    # print('jdata: ', jdata)
                    pt_lon = float(jdata["Location"]["original_lng"])
                    pt_lat = float(jdata["Location"]["original_lat"])
                    lonlats.append((pt_lon, pt_lat))
                    panoId = jdata["Location"]["panoId"]
                    heading = float(jdata["Projection"]["pano_yaw_deg"])
                    # print('yaw_list: ', yaw_list)
                    # print('Processing: ', panoId)
                    hasDownloaded = self.fileExists(saved_path, panoId + ".jpg")

                    if yaw_list[0] == 'json_only':
                        hasDownloaded = self.fileExists(saved_path, panoId + ".json")

                    if hasDownloaded:
                        print("File exists in forward(): ", saved_path, panoId + ".json")
                        return lonlats
                        # next_Json = self.getNextJson(jdata, pre_panoId)
                        # next_lon = next_Json["Location"]["original_lng"]
                        # next_lat = next_Json["Location"]["original_lat"]
                        # next_panoId = next_Json["Location"]["panoId"]
                        # step_cnt += 1
                        # next_pt = (next_lon, next_lat)
                        # pre_panoId = panoId
                        # print('step_cnt: ', step_cnt)
                        # continue

                    try:
                        json_name = os.path.join(saved_path, panoId + '.json')
                        if not self.fileExists(saved_path, panoId + ".json"):
                            with open(json_name, 'w') as f:
                                json.dump(jdata, f)
                    except Exception as e:
                        print("Error in go_along_road_forward() saving json file.", e, panoId)

                    print('Processing: ', panoId)

                    if yaw_list[0] == None:
                        # print('yaw_list : None', yaw_list)
                        self.getPanoJPGfrmPanoId(panoId, saved_path=saved_path, zoom=zoom)
                    elif yaw_list[0] == 'json_only':
                        # print("yaw_list: ", yaw_list)
                        pass
                    else:
                        # print("yaw_list: ", yaw_list)
                        self.getImageclipsfrmJson(jdata=jdata, yaw_list=yaw_list, pitch_list=pitch_list,
                                                  saved_path=saved_path)

                    # print('step:', step_cnt, jdata["Location"]["description"],  panoId)
                    # print(pt_lat, pt_lon)

                    next_Json = self.getNextJson(jdata, pre_panoId)
                    # print("next_Json: ", next_Json)

                    if next_Json == 0:
                        print("The road is dead end in Google Stree Map.")
                        return lonlats
                    else:
                        next_lon = next_Json["Location"]["original_lng"]
                        next_lat = next_Json["Location"]["original_lat"]
                        next_panoId = next_Json["Location"]["panoId"]

                        next_pt = (next_lon, next_lat)

                    step_cnt += 1

                    pre_panoId = panoId

                    # start_time = time.time()
                    # Cnt = 0
                    # Cnt_interval = 100

                    Cnt += 1
                    if Cnt % Cnt_interval == (Cnt_interval - 1):
                        print(
                            "Processing speed: {} steps / hour in forward(), pid: {}.".format(
                                int(Cnt / (time.time() - start_time + 0.001) * 3600), os.getpid()))

                except Exception as e:
                    print("Error in go_along_road_forward(), pid:", e, os.getpid())
                    print("Waiting for 5 seconds...")
                    time.sleep(5)
                    if len(pre_panoId) < 10:
                        return lonlats
                    next_Json = self.getNextJson(jdata, pre_panoId)
                    if next_Json == 0:
                        return lonlats

                    next_lon = next_Json["Location"]["original_lng"]
                    next_lat = next_Json["Location"]["original_lat"]
                    next_panoId = next_Json["Location"]["panoId"]
                    step_cnt += 1
                    next_pt = (next_lon, next_lat)
                    if panoId == 0:
                        return lonlats
                    pre_panoId = panoId
                    print('step_cnt: ', step_cnt)
                    continue
        except Exception as e:
            print("Error in go_along_road_forwakd(), function will restart.", e, os.getpid())
            self.go_along_road_forward(lon, lat, saved_path, yaw_list, pitch_list, steps, polygon,
                                       zoom)
        return lonlats

    def go_along_road_backward(self, lon, lat, saved_path, yaw_list=0, pitch_list=0, steps=0, polygon=None, zoom=5):
        lon = float(lon)
        lat = float(lat)
        if not self.point_in_polygon(Point((lon, lat)), polygon):
            print("Starting point is not in the polygon.")
            return

        if not isinstance(yaw_list, list):
            yaw_list = [yaw_list]

        step_cnt = 0
        next_pt = (lon, lat)
        next_panoId = self.getPanoIDfrmLonlat(lon, lat)[0]
        pre_panoId = ""

        lonlats = []
        pt_lon = lon
        pt_lat = lat

        if next_panoId == 0:
            return lonlats

        start_time = time.time()
        Cnt = 0
        Cnt_interval = 10

        try:

            while step_cnt < steps and self.point_in_polygon(Point(pt_lon, pt_lat), polygon):

                try:
                    # print(panoId)
                    jdata = self.getJsonfrmPanoID(next_panoId, saved_path=saved_path, dm=1)

                    # print('jdata: ', jdata)
                    pt_lon = float(jdata["Location"]["original_lng"])
                    pt_lat = float(jdata["Location"]["original_lat"])
                    lonlats.append((pt_lon, pt_lat))
                    panoId = jdata["Location"]["panoId"]
                    heading = float(jdata["Projection"]["pano_yaw_deg"])
                    # print('step:', step_cnt, jdata["Location"]["description"],  panoId)
                    # print(pt_lat, pt_lon)

                    hasDownloaded = self.fileExists(saved_path, panoId + ".jpg")
                    # print('yaw_list : None', yaw_list)
                    if yaw_list[0] == 'json_only':
                        hasDownloaded = self.fileExists(saved_path, panoId + ".json")
                        # print(yaw_list)

                    if hasDownloaded:
                        print("File exists in backward: ", os.path.join(saved_path, panoId + ".json"))
                        return lonlats
                        # print('yaw_list : None', yaw_list)
                        # next_Json = self.getLastJson(jdata, pre_panoId)
                        # next_lon = next_Json["Location"]["original_lng"]
                        # next_lat = next_Json["Location"]["original_lat"]
                        # next_panoId = next_Json["Location"]["panoId"]
                        # step_cnt += 1
                        # next_pt = (next_lon, next_lat)
                        # pre_panoId = panoId
                        # print('step_cnt: ', step_cnt)
                        # continue

                    try:
                        json_name = os.path.join(saved_path, panoId + '.json')
                        if not self.fileExists(saved_path, panoId + ".json"):
                            with open(json_name, 'w') as f:
                                json.dump(jdata, f)
                        else:
                            next_Json = self.getLastJson(jdata, pre_panoId)
                            next_lon = next_Json["Location"]["original_lng"]
                            next_lat = next_Json["Location"]["original_lat"]
                            next_panoId = next_Json["Location"]["panoId"]

                            next_pt = (next_lon, next_lat)

                            step_cnt += 1

                            pre_panoId = panoId

                            if len(pre_panoId) < 10:
                                return lonlats

                    except Exception as e:
                        print("Error in go_along_road_backward() saving json file.", e, panoId)

                    print('Processing: ', panoId)

                    if yaw_list[0] == None:
                        self.getPanoJPGfrmPanoId(panoId, saved_path=saved_path, zoom=zoom)
                    elif yaw_list[0] == 'json_only':
                        pass
                    else:

                        self.getImageclipsfrmJson(jdata=jdata, yaw_list=yaw_list, pitch_list=pitch_list,
                                                  saved_path=saved_path)

                    next_Json = self.getLastJson(jdata, pre_panoId)
                    # print("next_Json: ", next_Json)

                    if next_Json == 0:
                        print("The road is dead end in Google Stree Map.")
                        return lonlats
                    else:
                        next_lon = next_Json["Location"]["original_lng"]
                        next_lat = next_Json["Location"]["original_lat"]
                        next_panoId = next_Json["Location"]["panoId"]

                        next_pt = (next_lon, next_lat)

                    step_cnt += 1

                    pre_panoId = panoId

                    if len(pre_panoId) < 10:
                        return lonlats

                    if step_cnt % 100 == 0:
                        print("Step count:", step_cnt)

                        # start_time = time.time()
                        # Cnt = 0
                        # Cnt_interval = 100

                    Cnt += 1
                    if Cnt % Cnt_interval == (Cnt_interval - 1):
                        print(
                            "Processing speed: {} steps / hour in backward(), pid: {}".format(
                                int(Cnt / (time.time() - start_time + 0.001) * 3600), os.getpid()))


                except Exception as e:
                    print("Error in go_along_road_backward(), pid:", e, os.getpid())
                    print("Waiting for 5 seconds...")
                    time.sleep(5)
                    next_Json = self.getLastJson(jdata, pre_panoId)
                    if next_Json == 0:
                        return lonlats
                    next_lon = next_Json["Location"]["original_lng"]
                    next_lat = next_Json["Location"]["original_lat"]
                    next_panoId = next_Json["Location"]["panoId"]
                    step_cnt += 1
                    next_pt = (next_lon, next_lat)
                    if len(pre_panoId) < 10:
                        return lonlats

                    if panoId == 0:
                        return lonlats
                    pre_panoId = panoId

                    if len(pre_panoId) < 10:
                        return lonlats

                    print('step_cnt: ', step_cnt)
                    continue
        except Exception as e:
            print("Error in go_along_road_backwakd(), function will restart.", e, os.getpid())
            self.go_along_road_backward(lon, lat, saved_path, yaw_list, pitch_list, steps, polygon,
                                        zoom)

        return lonlats

    def getPanoJPGfrmArea(self, yaw_list, seed_pts, saved_path, boundary_vert, zoom=4):

        pts_cnt = len(seed_pts)
        polygon = self.formPolygon(boundary_vert)
        print(polygon)
        lonlats = []

        start_time = time.time()
        Cnt = 0
        Cnt_interval = 1

        try:
            while len(seed_pts) > 0:
                try:
                    pt = seed_pts.pop(0)
                    print(pt)
                    lonlats += (self.go_along_road_forward(pt[0], pt[1],
                                                           saved_path,
                                                           yaw_list=yaw_list, pitch_list=0,
                                                           steps=10000000000, polygon=polygon, zoom=zoom))

                    lonlats += (self.go_along_road_backward(pt[0], pt[1],
                                                            saved_path,
                                                            yaw_list=yaw_list, pitch_list=0,
                                                            steps=10000000000,
                                                            polygon=polygon, zoom=zoom))
                    print("Processed {} points, {} left.".format(pts_cnt - len(seed_pts), len(seed_pts)))
                    print("len(lonlats): ", len(lonlats))

                    # start_time = time.time()
                    # Cnt = 0
                    # Cnt_interval = 100

                    Cnt += 1
                    if Cnt % Cnt_interval == (Cnt_interval - 1):
                        print("Process speed: {} points / hour.".format(
                            int(Cnt / (time.time() - start_time + 0.001) * 3600)))
                except Exception as e:
                    print("Error in getPanoJPGfrmArea() loop:", e)
                    continue
        except Exception as e:
            print("Error in getPanoFrmArea(), will restart: ", e, os.getpid())
            self.getPanoJPGfrmArea(yaw_list, seed_pts, saved_path, boundary_vert)

    def getPanoJPGfrmArea_mp(self, yaw_list, seed_pts, saved_path, boundary_vert, zoom=4, Process_cnt=4):
        seed_pts_mp = mp.Manager().list()
        for seed in seed_pts:
            seed_pts_mp.append(seed)

        pool = mp.Pool(processes=Process_cnt)

        for i in range(Process_cnt):
            pool.apply_async(self.getPanoJPGfrmArea, args=(yaw_list, seed_pts_mp, saved_path, boundary_vert, zoom))
        pool.close()
        pool.join()

    def getPanoJPGfrmGap_mp(self, bearing_list, dangle_pair_pts, saved_path, step_nums=10, Process_cnt=4):
        seed_pts_mp = mp.Manager().list()
        for seed in seed_pts:
            seed_pts_mp.append(seed)

        pool = mp.Pool(processes=Process_cnt)

        for i in range(Process_cnt):
            pool.apply_async(self.getPanoJPGfrmArea, args=(yaw_list, seed_pts_mp, saved_path, boundary_vert, zoom))
        pool.close()
        pool.join()

    def shootLonlat(self, ori_lon, ori_lat, saved_path='', views=1, prefix='', suffix='', width=1024, height=768, pitch=0):

        # panoid, lon, lat = self.getPanoIDfrmLonlat(ori_lon, ori_lat)
        try:
            jdata = self.getPanoJsonfrmLonat(ori_lon, ori_lat)

        except Exception as e:
            print("Error in getting json in shootLonlat():", e)
        if 'Location' in jdata:
            panoid = jdata['Location']['panoId']
            lon = jdata['Location']['original_lng']
            lat = jdata['Location']['original_lat']
        else:
            print("No Location in the Panojson file.")
            return 0, 0

        if panoid == 0:
            print("No PanoID return for location: ", ori_lon, ori_lat)
            return 0, 0
        lon = float(lon)
        lat = float(lat)

        # print('lon/lat in panorama:', lon, lat)
        heading = self.getDegreeOfTwoLonlat(lat, lon, ori_lat, ori_lon)

        # print(idx, 'Heading angle between tree and panorama:', heading)
        # f.writelines(f"{ID},{ACCTID}{ori_lon},{ori_lat},{lon},{lat},{heading}" + '\n')
        image, jpg_name = self.getImagefrmAngle(lon, lat, saved_path=saved_path, prefix=str(prefix) + panoid,
                              pitch=pitch, yaw=heading)

        if views == 1:
            return image, jpg_name

        if views == 3:
            images = []
            jpg_names = []
            images.append(image)
            try:
                links = jdata["Links"]
            except Exception as e:
                print("Error in shootLonlat() getting Links: ", e)
                return 0, 0

            for link in links:
                try:
                    # print("link:", link)
                    panoId1 = link['panoId']
                    jdata1 = self.getJsonfrmPanoID(panoId1)
                    # print("jdata1: ", jdata1['Location']['panoId'])
                    lon = jdata1['Location']['original_lng']
                    lat = jdata1['Location']['original_lat']
                    lon = float(lon)
                    lat = float(lat)
                    heading = self.getDegreeOfTwoLonlat(lat, lon, ori_lat, ori_lon)
                    image, jpg_name = self.getImagefrmAngle(lon, lat, saved_path=saved_path, prefix=prefix,
                                          pitch=pitch, yaw=heading)
                    images.append(image)
                    jpg_names.append(jpg_name)
                except Exception as e:
                    print("Error in processing links in shootLonlat():", e)
                    continue
            # if saved_path != "":
            return images, jpg_names

    def getJsonfrmPanoID(self, panoId, dm=0, saved_path=''):
        url = f"http://maps.google.com/cbk?output=json&panoid={panoId}&dm={dm}"
        try:
            r = requests.get(url)
            jdata = r.json()
            if saved_path != '':
                try:
                    with open(os.path.join(saved_path, panoId + '.json'), 'w') as f:
                        json.dump(jdata, f)
                except Exception as e:
                    print("Error in getJsonfrmPanoID() saving json file.", e, panoId)
            return jdata
        except Exception as e:
            print("Error in getJsonfrmPanoID():", e)
            return 0

    def shootLonlats(self, ori_lonlats, saved_path, views=1, suffix='', width=1024, height=768, pitch=0):
        # prepare for calculating processing speed
        start_time = time.time()
        ini_len = len(ori_lonlats)
        Cnt = 0
        Cnt_interval = 100
        while len(ori_lonlats) > 0:
            ori_lon, ori_lat, prefix = ori_lonlats.pop(0)
            try:
                self.shootLonlat(ori_lon, ori_lat, saved_path=saved_path, views=views, prefix=prefix, width=width,
                                 height=height, pitch=0)
                # calculate processing speed
                Cnt += 1
                if Cnt % Cnt_interval == (Cnt_interval - 1):
                    speed_pt_hour = int((ini_len - len(ori_lonlats)) / (time.time() - start_time + 0.001) * 3600)
                    print("Process speed: {} points / hour by all processes.".format(speed_pt_hour))
                    print("Processed {} / {},  {:.2%}, {:.2f} hours left.".format(ini_len - len(ori_lonlats), ini_len, (
                                ini_len - len(ori_lonlats)) / ini_len, (len(ori_lonlats) / speed_pt_hour)))

            except Exception as e:
                print("Error in shootLonlats():", e, ori_lon, ori_lat, prefix)

    def shootLonlats_mp(self, ori_lonlats_mp, saved_path, Process_cnt=4, views=1, suffix='', width=1024, height=768,
                        pitch=0):
        pool = mp.Pool(processes=Process_cnt)
        for i in range(Process_cnt):
            pool.apply_async(self.shootLonlats, args=(ori_lonlats_mp, saved_path, views))
        pool.close()
        pool.join()

    def getImageCirclefrmLonlat(self, lon: float, lat: float, saved_path='Panos', prefix='', suffix='', width=1024,
                                height=768, pitch=0, road_compassA=0, interval=90):
        # w maximum: 1024
        # h maximum: 768
        # FOV should be 90, cannot be changed
        # interval: degree, not rad
        interval = abs(interval)
        interval = max(interval, 1)

        img_cnt = math.ceil(360 / interval)
        for i in range(img_cnt):
            yaw = road_compassA + i * interval
            self.getImagefrmAngle(lon, lat, saved_path, prefix, suffix, width, height, pitch, yaw)

    def getImage4DirectionfrmLonlat(self, lon: float, lat: float, saved_path='Panos', prefix='', suffix='', width=1024,
                                    height=768, pitch=0, road_compassA=0):
        # w maximum: 1024
        # h maximum: 768
        # FOV should be 90, cannot be changed
        # interval: degree, not rad
        suffix = str(suffix)
        if suffix != '':
            suffix = '_' + suffix

        # img_cnt = math.ceil(360/interval)
        names = ['F', 'R', 'B', 'L']  # forward, backward, left, right
        # interval = math.ceil(360 / len(names))
        for idx, name in enumerate(names):
            yaw = road_compassA + idx * 90
            # print('idx: ', idx)
            # print('name:', name)
            # print('yaw: ', yaw)
            self.getImagefrmAngle(lon, lat, saved_path, prefix, name + suffix, width, height, pitch, yaw)

    def getImage8DirectionfrmLonlat(self, lon: float, lat: float, saved_path='Panos', prefix='', suffix='', width=1024,
                                    height=768, pitch=0, road_compassA=0):
        # w maximum: 1024
        # h maximum: 768
        # FOV should be 90, cannot be changed
        # interval: degree, not rad
        suffix = str(suffix)
        if suffix != '':
            suffix = '_' + suffix

        # img_cnt = math.ceil(360/interval)
        # names = ['F', 'R', 'B', 'L']  # forward, backward, left, right
        names = ['F', 'FR', 'R', 'RB', 'B', 'BL', 'L', 'LF']  # forward, backward, left, right
        interval = math.ceil(360 / len(names))
        for idx, name in enumerate(names):
            yaw = road_compassA + idx * interval
            # print('idx: ', idx)
            # print('name:', name)
            # print('yaw: ', yaw)
            self.getImagefrmAngle(lon, lat, saved_path, prefix, name + suffix, width, height, pitch, yaw)

    def getImage4DirectionfrmLonlats(self, list_lonlat, saved_path='Panos', prefix='', suffix='', width=1024,
                                     height=768, pitch=0, road_compassA=0):
        # print(len(list_lonlat))

        start_time = time.time()
        Cnt = 0
        Cnt_interval = 100
        origin_len = len(list_lonlat)

        while len(list_lonlat) > 0:
            try:
                # print(list_lonlat.pop(0))
                lon, lat, id, prefix, road_compassA = list_lonlat.pop(0)
                prefix = str(id)

                print('Current row :', id)
                self.getImage8DirectionfrmLonlat(lon, lat, saved_path, prefix, suffix, width, height, pitch,
                                                 road_compassA)
                current_len = len(list_lonlat)
                Cnt = origin_len - current_len
                if Cnt % Cnt_interval == (Cnt_interval - 1):
                    print(
                        "Prcessed {} / {} items. Processing speed: {} points / hour.".format(Cnt, origin_len, int(
                            Cnt / (time.time() - start_time + 0.001) * 3600)))
            except Exception as e:
                print("Error in getImage4DirectionfrmLonlats(): ", e, id)
                current_len = len(list_lonlat)
                continue

    def getImage4DirectionfrmLonlats_mp(self, list_lonlat_mp, saved_path='Panos', Process_cnt=6, prefix='', suffix='',
                                        width=1024, height=768, pitch=0, road_compassA=0):
        # statuses = []      # succeeded: 1; failed: 0
        pool = mp.Pool(processes=Process_cnt)

        for i in range(Process_cnt):
            pool.apply_async(self.getImage4DirectionfrmLonlats,
                             args=(list_lonlat_mp, saved_path, prefix, suffix, width, height, pitch, road_compassA))
        pool.close()
        pool.join()

    def castesian_to_shperical_pp(self, rowcols, w, h, fov):  # yaw: set the heading, pitch

        f = (0.5 * w) / math.tan(fov * 0.5)
        # print("f:", f)
        if not isinstance(rowcols, list):
            rowcols = [rowcols]

        theta_phis = []
        # print("type of colrows:", colrows)
        # plt_x = []
        # plt_y = []
        # print("len of : colrows", len(colrows))

        resolution_angle = fov / w
        print('resolution: ', resolution_angle)

        for rowcol in rowcols:
            # print('colrow, len of colrow: ', colrow, len(colrow))
            row = rowcol[0]
            col = rowcol[1]
            new_rowcols = 0
            # print('col, row:', col, row)
            x = col - w * 0.5
            y = h * 0.5 - row
            z = f

            phi = x * resolution_angle

            theta = y * resolution_angle

            theta_phis.append((theta, phi))

        # print('theta_phis:', theta_phis)
        # print(x, y, z, theta + heading, phi + pitch, pi)
        # plt.scatter(plt_x, plt_y)
        # plt.show()
        print("theta_phis[0]:", theta_phis[0])

        if len(theta_phis) == 1:
            return theta_phis[0]
        else:
            return theta_phis

    # def castesian_to_shperical0(self, rowcols, w, h, fov):  # yaw: set the heading, pitch
    #     fov = radians(fov)
    #     f = (0.5 * w) / math.tan(fov * 0.5)
    #     # print("f:", f)
    #     if not isinstance(rowcols, list):
    #         rowcols = [rowcols]
    #
    #     theta_phis = []
    #     # print("type of colrows:", colrows)
    #     # plt_x = []
    #     # plt_y = []
    #     # print("len of : colrows", len(colrows))
    #
    #     for colrow in rowcols:
    #         # print('colrow, len of colrow: ', colrow, len(colrow))
    #         row = colrow[0]
    #         col = colrow[1]
    #         new_rowcols = 0
    #         # print('col, row:', col, row)
    #         x = col - w * 0.5
    #         y = h * 0.5 - row
    #         z = f
    #
    #         # plt_x.append(x)
    #         # print('x, y, z, w, h:', x, y, z, w, h)
    #
    #         #  theta = atan(y / z)
    #         # print('theta:', theta)
    #         # if y < 0:
    #         #     theta = theta * -1
    #         # print('theta:', theta)
    #
    #         # theta = atan(y / z)
    #
    #         # theta = acos(z / sqrt(x * x + y * y + z * z))
    #         # if y < 0:
    #         #     theta = -1 * theta
    #
    #         # phi = acos(x / sqrt(x * x + y * y))
    #         # if y < 0:
    #         #     #phi = 2 * pi - phi
    #         #     phi = -1 * phi
    #
    #         phi = acos(z / sqrt(x * x + z * z))
    #         if x < 0:
    #             phi = -phi
    #         theta = atan(y / sqrt(x * x + z * z))
    #
    #         # phi = atan(x / y)
    #
    #         theta = degrees(theta)
    #         phi = degrees(phi)
    #         # plt_y.append(phi)
    #
    #         theta_phis.append((theta, phi))
    #
    #         # print("col, row, x, y, z, theta, phi:")
    #         # print(col, row, x, y, z, theta, phi)
    #     # print('theta_phis:', theta_phis)
    #     # print(x, y, z, theta + heading, phi + pitch, pi)
    #     # plt.scatter(plt_x, plt_y)
    #     # plt.show()
    #     if len(theta_phis) == 1:
    #         return theta_phis[0]
    #     else:
    #         return theta_phis

    def clip_pano(self, np_img, theta=0, phi=0, w=1024, h=768, fov=90):  # fov < 120
        if fov > 120:
            fov = 120
            print("Fov should be < 120 degree.")

        print('np_img.shape:', np_img.shape)
        pano_w = np_img.shape[1]
        pano_h = np_img.shape[0]

        print('pano_w, pano_h:', pano_w, pano_h)

        cols = [x for x in range(w)]
        rows = [x for x in range(h)]

        print('cols:', cols)
        print('rows:', rows)

        cols_mesh, rows_mesh = np.meshgrid(cols, rows)
        print('len of cols_mesh:', len(cols_mesh))

        print('len of rows_mesh:', len(rows_mesh))
        print('size of rows_mesh:', rows_mesh.size)
        # print(rows_mesh[:5])
        # print(cols_mesh[:5])

        # zl = zip(cols_mesh, rows_mesh)
        #
        # print('zip of cols_mesh:', zl)
        fl = cols_mesh.flatten()
        print(len(fl))
        print(len(rows_mesh.flatten()))
        print(len([w] * w * h))

        thetas_phis = map(self.colrow_to_spherial, cols_mesh.flatten(), rows_mesh.flatten(), [w] * w * h, [h] * w * h,
                          [fov] * w * h)
        thetas_phis = list(thetas_phis)
        # print('theta_phis:', thetas_phis)
        print('theta_phis:', thetas_phis[0])
        res = pi / pano_h  # angle resolution of pano

        print('res: ', res)

        img = Image.new('RGB', (w, h))
        pixels = img.load()
        np_pixels = Image.fromarray(np_img).load()
        for r in range(h):
            for c in range(w):
                theta_pixel = thetas_phis[r * w + c][0] + theta
                phi_pixel = thetas_phis[r * w + c][1] + phi
                if phi_pixel > pi:
                    phi_pixel = phi_pixel - 2 * pi
                if phi_pixel < -pi:
                    phi_pixel = phi_pixel + 2 * pi

                # have not processed theta

                p_row = int(pano_h / 2 - theta_pixel / res)
                p_col = int(phi_pixel / res + pano_w / 2)
                # print(p_col, p_row)
                p_row = min(p_row, pano_h - 1)
                p_col = min(p_col, pano_w - 1)
                pixels[c, r] = np_pixels[p_col, p_row]
                # print(np_pixels[p_col, p_row])
        # img.show()

        return img

    def clip_pano2(self, theta0, phi0, fov_h, fov_v, width, img):  # fov < 120
        """
          theta0 is pitch
          phi0 is yaw
          render view at (pitch, yaw) with fov_h by fov_v
          width is the number of horizontal pixels in the view
          """
        m = np.dot(yrotation(phi0), xrotation(theta0))

        (base_height, base_width, _) = img.shape

        height = int(width * np.tan(fov_v / 2) / np.tan(fov_h / 2))

        new_img = np.zeros((height, width, 3), np.uint8)

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
        return new_img




class GSV_depthmap(object):

    # def getPanoIdDepthmapfrmLonlat(self, lon, lat, dm=1, saved_path='', prefix='', suffix=''):
    #     url = f''
    #     r = requests.get('url')
    #     print(r)

    def getJsonDepthmapfrmLonlat(self, lon, lat, dm=1, saved_path='', prefix='', suffix=''):
        prefix = str(prefix)
        suffix = str(suffix)
        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix
        url = f'http://maps.google.com/cbk?output=json&ll={lat},{lon}&dm={dm}'
        # print(url)
        try:
            r = requests.get(url)
            jdata = r.json()
            # str_dm = jdata['model']['depth_map']

            if saved_path == '':
                return jdata
            else:
                mapname = os.path.join(saved_path, prefix + jdata['Location']['original_lng'] + '_' + jdata['Location'][
                    'original_lat'] + '_' + jdata['Location']['panoId'] + suffix + '.json')
                with open(mapname, 'w') as f:
                    json.dump(jdata, f)


        except Exception as e:
            print("Error in getPanoIdDepthmapfrmLonlat():", str(e))
            print(url)

    def getJsonDepthmapsfrmLonlats(self, lonlat_list, dm=1, saved_path='', prefix='', suffix=''):
        start_time = time.time()
        Cnt = 0
        Cnt_interval = 1000
        origin_len = len(lonlat_list)

        while len(lonlat_list) > 0:
            lon, lat, id, idx = lonlat_list.pop(0)
            prefix = id
            prefix = str(prefix)
            suffix = str(suffix)
            if prefix != "":
                prefix += '_'
            if suffix != "":
                suffix = '_' + suffix
            url = f'http://maps.google.com/cbk?output=json&ll={lat},{lon}&dm={dm}'
            print("Current row:", idx)
            try:
                r = requests.get(url)
                jdata = r.json()
                # str_dm = jdata['model']['depth_map']

                mapname = os.path.join(saved_path, prefix + jdata['Location']['original_lng'] + '_' + jdata['Location'][
                    'original_lat'] + '_' + jdata['Location']['panoId'] + suffix + '.json')

                with open(mapname, 'w') as f:
                    json.dump(jdata, f)

                current_len = len(lonlat_list)
                Cnt = origin_len - current_len
                if Cnt % Cnt_interval == (Cnt_interval - 1):
                    print(
                        "Prcessed {} / {} items. Processing speed: {} points / hour.".format(Cnt, origin_len, int(
                            Cnt / (time.time() - start_time + 0.001) * 3600)))


            except Exception as e:
                print("Error in getJsonDepthmapsfrmLonlats():", str(e))
                print(url)
                continue

    def getJsonDepthmapsfrmLonlats_mp(self, lonlat_list, dm=1, saved_path='', prefix='', suffix='', Process_cnt=4):

        try:
            pool = mp.Pool(processes=Process_cnt)
            for i in range(Process_cnt):
                pool.apply_async(self.getJsonDepthmapsfrmLonlats, args=(lonlat_list, dm, saved_path, prefix, suffix))
            pool.close()
            pool.join()


        except Exception as e:
            print("Error in getJsonDepthmapsfrmLonlats_mp():", str(e))

    def lonlat2WebMercator(self, lon, lat):
        # return transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), lon, lat)
        return transform(Proj(init='epsg:4326'), Proj(init='epsg:2824'), lon, lat)

    def WebMercator2lonlat(self, X, Y):
        return transform(Proj(init='epsg:3857'), Proj(init='epsg:4326'), X, Y)

    def parse(self, b64_string):
        # fix the 'inccorrect padding' error. The length of the string needs to be divisible by 4.
        b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
        # convert the URL safe format to regular format.
        data = b64_string.replace("-", "+").replace("_", "/")

        data = base64.b64decode(data)  # decode the string
        data = zlib.decompress(data)  # decompress the data
        return np.array([d for d in data])

    def parseHeader(self, depthMap):
        return {
            "headerSize": depthMap[0],
            "numberOfPlanes": self.getUInt16(depthMap, 1),
            "width": self.getUInt16(depthMap, 3),
            "height": self.getUInt16(depthMap, 5),
            "offset": self.getUInt16(depthMap, 7),
        }

    def get_bin(self, a):
        ba = bin(a)[2:]
        return "0" * (8 - len(ba)) + ba

    def getUInt16(self, arr, ind):
        a = arr[ind]
        b = arr[ind + 1]
        return int(self.get_bin(b) + self.get_bin(a), 2)

    def getFloat32(self, arr, ind):
        return self.bin_to_float("".join(self.get_bin(i) for i in arr[ind: ind + 4][::-1]))

    def bin_to_float(self, binary):
        return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

    def parsePlanes(self, header, depthMap):
        indices = []
        planes = []
        n = [0, 0, 0]

        for i in range(header["width"] * header["height"]):
            indices.append(depthMap[header["offset"] + i])

        for i in range(header["numberOfPlanes"]):
            byteOffset = header["offset"] + header["width"] * header["height"] + i * 4 * 4
            n = [0, 0, 0]
            n[0] = self.getFloat32(depthMap, byteOffset)
            n[1] = self.getFloat32(depthMap, byteOffset + 4)
            n[2] = self.getFloat32(depthMap, byteOffset + 8)
            d = self.getFloat32(depthMap, byteOffset + 12)
            planes.append({"n": n, "d": d})

        return {"planes": planes, "indices": indices}

    def computeDepthMap(self, header, indices, planes):

        v = [0, 0, 0]
        w = header["width"]
        h = header["height"]

        depthMap = np.empty(w * h)

        sin_theta = np.empty(h)
        cos_theta = np.empty(h)
        sin_phi = np.empty(w)
        cos_phi = np.empty(w)

        for y in range(h):
            theta = (h - y) / h * np.pi  # original
            # theta = y / h * np.pi  # huan
            sin_theta[y] = np.sin(theta)
            cos_theta[y] = np.cos(theta)

        for x in range(w):
            phi = x / w * 2 * np.pi  # + np.pi / 2
            sin_phi[x] = np.sin(phi)
            cos_phi[x] = np.cos(phi)

        for y in range(h):
            for x in range(w):
                planeIdx = indices[y * w + x]

                # Origninal
                # v[0] = sin_theta[y] * cos_phi[x]
                # v[1] = sin_theta[y] * sin_phi[x]

                # Huan
                v[0] = sin_theta[y] * sin_phi[x]
                v[1] = sin_theta[y] * cos_phi[x]
                v[2] = cos_theta[y]

                if planeIdx > 0:
                    plane = planes[planeIdx]
                    t = np.abs(plane["d"] / (v[0] * plane["n"][0] + v[1] * plane["n"][1] + v[2] * plane["n"][2]))
                    # original
                    #     depthMap[y * w + (w - x - 1)] = t
                    # else:
                    #     depthMap[y * w + (w - x - 1)] = 0

                    # huan
                    if t < 100:
                        depthMap[y * w + x] = t
                    else:
                        depthMap[y * w + x] = 9999999
                else:
                    depthMap[y * w + x] = 9999999

        return {"width": w, "height": h, "depthMap": depthMap}

    def getDepthmapfrmJson(self, jdata):
        try:
            depthMapData = self.parse(jdata['model']['depth_map'])
            # parse first bytes to describe data
            header = self.parseHeader(depthMapData)
            # parse bytes into planes of float values
            data = self.parsePlanes(header, depthMapData)
            # compute position and values of pixels
            depthMap = self.computeDepthMap(header, data["indices"], data["planes"])
            return depthMap
        except Exception as e:
            print("Error in getDepthmapfrmJson():", e)

    def saveDepthmapImage(self, depthMap, saved_file):
        im = depthMap["depthMap"]

        # print(im)
        im[np.where(im == max(im))[0]] = 0
        # if min(im) < 0:
        #     im[np.where(im < 0)[0]] = 0
        im = im.reshape((depthMap["height"], depthMap["width"]))  # .astype(int)
        # display image
        img = PIL.Image.fromarray(im)
        # img.save(saved_file.replace(".tif", 'noflip.tif'))
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(saved_file)

        # print(im)

    def getPointCloud(self, theta_phis_in_pano, heading_of_thumb, pitch_of_thumb, \
                      depthmap, cameraLon, cameraLat,
                      cameraH, heading_of_pano, pitch_of_pano):
        try:
            print('heading_of_thumb:', math.degrees(heading_of_thumb))
            print('pitch_of_thumb:', math.degrees(pitch_of_thumb))
            results = []

            # if not isinstance(theta_phis_in_pano, list):
            #     theta_phis_in_thumb = [theta_phis_in_pano]

            results = theta_phis_in_pano
            print('results shape: ', results.shape, results[0])
            heading_of_thumb = float(heading_of_thumb)
            pitch_of_thumb = float(pitch_of_thumb)
            cameraLon = float(cameraLon)
            cameraLat = float(cameraLat)
            cameraH = float(cameraH)
            heading_of_pano = float(heading_of_pano)
            pitch_of_pano = float(pitch_of_pano)
            print('results shape: ', results.shape, results[0])
            # for row in theta_phis_in_thumb:
            #     print("theta_phis_in_thumb (before adding thumb_heading): ", row)

            # theta_phis_in_thumb = [tuple([row[0] - pitch_of_thumb, row[1] + heading_of_thumb]) for row in theta_phis_in_thumb]

            # results = np.array(theta_phis_in_thumb)
            print('results shape: ', results.shape, results[0])
            results = np.concatenate((results, theta_phis_in_pano), axis=1)

            print('results shape: ', results.shape, results[0])

            # for row in theta_phis_in_thumb:
            #     print("theta_phis_in_thumb (after adding thumb_heading): ", row)
            #
            # for row in theta_phis_in_thumb:
            #     print("theta_phis_in_thumb (after adding pano_heading): ", row[1] + heading_of_pano)

            # print("row: ", row)
            points3D = np.zeros((len(theta_phis_in_pano), 4))
            print('depthmap[width]: ', depthmap['width'])
            print('depthmap[height]:', depthmap['height'])
            # print('theta_phis_in_thumb[0]: ', math.degrees(theta_phis_in_thumb[0][0]), math.degrees(theta_phis_in_thumb[0][1]))
            cnt = 0
            cameraX, cameraY = self.lonlat2WebMercator(cameraLon, cameraLat)
            print('cameraX, cameraY:', cameraX, cameraY)
            # heading_of_pano = radians(heading_of_pano)
            # pitch_of_pano = radians(pitch_of_pano)
            dempth_image = np.array(depthmap['depthMap']).reshape(256, 512)
            # plt.imshow(dempth_image)
            # plt.show()

            # dempth_image = scipy.ndimage.zoom(dempth_image, 4, order=3)
            # print("Shape of dempth_image: ", dempth_image.shape)
            # print('dempth_image: ', dempth_image)
            # plt.imshow(dempth_image)
            # plt.show()

            dm_h, dm_w = dempth_image.shape
            print("dempth_image.shape: ", dempth_image.shape, dm_h, dm_w)

            grid_col = np.linspace(-pi, pi, dm_w)
            grid_row = np.linspace(pi , 0, dm_h)
            # gridxx = np.arange(512)
            # gridyy = np.arange(256)
            # gridxx, gridyy = np.meshgrid(grid_col, grid_row)
            # print("shapes: gridx, gridy", grid_col.shape, grid_row.shape, dempth_image.shape)
            # print("types: gridx, dempth_image", (gridx), (dempth_image))
            # print(gridx)
            # print(gridy)
            # print('dempth_image: ', dempth_image)

            # fig = plt.figure(figsize=(9, 6))
            # Draw sub-graph1
            # ax = plt.subplot(1, 2, 1, projection='3d')
            # surf = ax.plot_surface(gridxx, gridyy, dempth_image, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('f(x, y)')
            # plt.colorbar(surf, shrink=0.5, aspect=5)  # 标注
            # plt.show()

            # 二维插值
            interp = interpolate.interp2d(grid_col, grid_row, dempth_image,
                                          kind='linear')  # 'cubic' will distord the distance.
            print('theta_phis_in_pano.shape: ', theta_phis_in_pano.shape)
            print('theta_phis_in_pano[:, 1]: ', theta_phis_in_pano[:, 1])
            print('theta_phis_in_pano[:, 0]: ', theta_phis_in_pano[:, 0])

            # distances = interp(theta_phis_in_pano[-10:, 1], theta_phis_in_pano[-10:, 0])
            # print('distances[0]: ', distances)
            idx = 0

            for ray in theta_phis_in_pano:
                # if ray[1] < 0:
                #     print('ray (degree):', ray)

                theta1 = ray[1]
                theta =theta1-math.pi/2.0

                phi1 = ray[0] -math.pi
                phi = phi1- (heading_of_thumb-heading_of_pano)

                if phi>math.pi:
                    phi = phi-math.pi*2.0
                elif phi<-math.pi:
                    phi = phi +math.pi*2.0

                # if phi > pi:
                #     phi = phi - pi

                # x = phi / pi * depthmap['width'] / 2
                # y = theta / (pi / 2) * depthmap['height'] / 2
                # row = int(depthmap['height'] / 2 - y)
                # # row = min(row - 5, 0)
                # col = int(x + depthmap['width'] / 2)

                # x = phi / pi * dm_w / 2
                # y = theta / (pi/2) * dm_h / 2
                # row = int(dm_h / 2 - y)
                # col = int(x + dm_w / 2)

                #

                # distance0 = depthmap['depthMap'][depthmap['width'] * row + col]
                # distance = interp(phi, theta - pitch_of_pano)[0]
                # distance = interp(phi,theta)[0]

                distance = interp(phi1, theta1)[0]

                # # distance = interp(phi, min(theta + 0.05, 0) )[0]
                # if distance > 2 and distance < 20:
                #     print("distance：", idx, ray, distance)
                # distance = distance * 2
                # if distance < 0:
                #     print("distance < 0：", distance, distance0, col, row)
                # print(type(distance))
                # print(distance.shape)
                # print(distance.size)
                # distance = dempth_image[row][col]
                # if idx % 1500 == 0:
                #     print('distance:', theta, phi, interp(phi, theta))
                #     print('distance:', x, y, col, row, distance)
                #     print('distance:', dempth_image[row][col])

                # if distance > 1:
                # pointX = cameraX + distance * cos(theta + pitch_of_pano) * sin(phi + heading_of_pano)
                # pointY = cameraY + distance * cos(theta + pitch_of_pano) * cos(phi + heading_of_pano)
                # pointZ = cameraH + distance * sin(theta + pitch_of_pano)
                # print("pitch_of_pano:", pitch_of_pano)

                # pointX = cameraX + distance * cos(theta + pitch_of_pano) * sin(phi + heading_of_pano)
                # pointY = cameraY + distance * cos(theta + pitch_of_pano) * cos(phi + heading_of_pano)
                # pointZ = cameraH + distance * sin(theta + pitch_of_pano)


                pointX =  distance * cos(theta) * sin(phi)
                pointY =  distance * cos(theta ) * cos(phi )
                pointZ =  distance * sin(theta )

                # else:
                #     pointX = 0
                #     pointY = 0
                #     pointZ = 0

                points3D[idx] = np.array([pointX, pointY, pointZ, distance])
                idx += 1

                # print("distance * cos(theta):",distance * cos(theta))
                # #
                #
                # print('point_lat, point_lon: ', self.WebMercator2lonlat(pointX, pointY))
                # print("pointX, pointY, pointZ, distance: ", pointX, pointY, pointZ, distance)
                # print("d_pointX, d_pointY, d_pointZ: ", distance * cos(theta) * sin(phi + heading_of_pano), distance * cos(theta) * cos(phi + heading_of_pano), distance * sin(theta))
                # if cnt < 3:
                #     print('theta, phi, x, y, row, col:', theta, phi, x, y, row, col)
                # cnt += 1
            # results = np.concatenate((results, np.array(points3D)), axis=1)
           #transform to 3d coordinates

            pointsCloud_xyz = points3D[:, :3]

            def RawPointsCloud2Camera3D( pointsCloud, alpha, beta, gamma):

                pointsCloud = self.rotate_z(-gamma).dot(self.rotate_x(-beta)).dot(self.rotate_z(-alpha)).dot(
                    pointsCloud.T).T

                return pointsCloud


            # pointsCloud_xyz[:] = self.RawPointsCloud2Camera3D(pointsCloud_xyz,pano_yaw,tilt_pitch,tilt_yaw)
            # pointsCloud_xyz[:] = self.RawPointsCloud2Camera3D(pointsCloud_xyz, pano_yaw, 0, 0)
            pointsCloud_xyz[:] = RawPointsCloud2Camera3D(pointsCloud_xyz, heading_of_thumb, pitch_of_thumb, 0)

            pointsCloud_xyz[:] = pointsCloud_xyz + np.array([cameraX, cameraY, cameraH])

            print('final points3D shape : ', points3D[0:5], len(points3D))
            return points3D  # ， results
        except Exception as e:
            print("Error in getPointCloud():", e)

    # DO not use it!!
    def getPointCloud2(self, theta_phis_in_pano, heading_of_thumb, pitch_of_thumb, depthmap, cameraLon, cameraLat,
                       cameraH, heading_of_pano, pitch_of_pano, sub_w = 1024*2, sub_h=768*2):
        try:
            # print('heading_of_thumb:', math.degrees(heading_of_thumb))


            # if not isinstance(theta_phis_in_pano, list):
            #     theta_phis_in_thumb = [theta_phis_in_pano]

            results = theta_phis_in_pano
            # print('results shape: ', results.shape, results[0])
            heading_of_thumb = float(heading_of_thumb)
            pitch_of_thumb = float(pitch_of_thumb)
            cameraLon = float(cameraLon)
            cameraLat = float(cameraLat)
            cameraH = float(cameraH)
            heading_of_pano = float(heading_of_pano)
            pitch_of_pano = float(pitch_of_pano)
            # print('results shape: ', results.shape, results[0])
            # for row in theta_phis_in_thumb:
            #     print("theta_phis_in_thumb (before adding thumb_heading): ", row)


            # print('results shape: ', results.shape, results[0])
            results = np.concatenate((results, theta_phis_in_pano), axis=1)

            # print('results shape: ', results.shape, results[0])

            # for row in theta_phis_in_thumb:
            #     print("theta_phis_in_thumb (after adding thumb_heading): ", row)
            #
            # for row in theta_phis_in_thumb:
            #     print("theta_phis_in_thumb (after adding pano_heading): ", row[1] + heading_of_pano)

            # print("row: ", row)
            points3D = np.zeros((len(theta_phis_in_pano), 4))
            # print('depthmap[width]: ', depthmap['width'])
            # print('depthmap[height]:', depthmap['height'])
            # print('theta_phis_in_thumb[0]: ', math.degrees(theta_phis_in_thumb[0][0]), math.degrees(theta_phis_in_thumb[0][1]))
            cnt = 0
            cameraX, cameraY = self.lonlat2WebMercator(cameraLon, cameraLat)
            # print('cameraX, cameraY:', cameraX, cameraY)
            # heading_of_pano = radians(heading_of_pano)
            # pitch_of_pano = radians(pitch_of_pano)
            dempth_image = np.array(depthmap['depthMap']).reshape(256, 512)
            # plt.imshow(dempth_image)
            # plt.show()

            # dempth_image = scipy.ndimage.zoom(dempth_image, 4, order=3)
            # print("Shape of dempth_image: ", dempth_image.shape)
            # print('dempth_image: ', dempth_image)
            # plt.imshow(dempth_image)
            # plt.show()

            dm_h, dm_w = dempth_image.shape
            # print("dempth_image.shape: ", dempth_image.shape, dm_h, dm_w)

            grid_col = np.linspace(-pi, pi, dm_w)
            grid_row = np.linspace(pi / 2, -pi / 2, dm_h)



            # 二维插值
            interp = interpolate.interp2d(grid_col, grid_row, dempth_image,
                                          kind='linear')  # 'cubic' will distord the distance.
            # print('theta_phis_in_pano.shape: ', theta_phis_in_pano.shape)
            # print('theta_phis_in_pano[:, 1]: ', theta_phis_in_pano[:, 1])
            # print('theta_phis_in_pano[:, 0]: ', theta_phis_in_pano[:, 0])

            min_theta = np.min(theta_phis_in_pano[:, 1])
            max_theta = np.max(theta_phis_in_pano[:, 1])

            min_phi = np.min(theta_phis_in_pano[:, 0])
            max_phi = np.max(theta_phis_in_pano[:, 0])



            new_grid_col = np.linspace(min_phi, max_phi, sub_w)
            new_grid_row = np.linspace(max_theta, min_theta, sub_h)

            sub_depthmap = interp(new_grid_col, new_grid_row)

            # distances = interp(theta_phis_in_pano[-10:, 1], theta_phis_in_pano[-10:, 0])
            resolution_phi = (max_phi - min_phi) / sub_w
            resolution_theta = (max_theta - min_theta) / sub_h
            col_rows = (theta_phis_in_pano - np.array([min_phi, min_theta]))/ np.array(resolution_phi, resolution_theta)
            col_rows = np.rint(col_rows).astype(int)
            bou = np.array([sub_w - 1, sub_h - 1])
            col_rows = np.where(col_rows > bou, bou, col_rows)
            distances = sub_depthmap[col_rows[:, 1], col_rows[:, 0]]

            # distances = distances[distances < 20]
            # distances = distances[distances > 0]
            # pointX = distances * np.cos(theta_phis_in_pano[:, 1]) * np.sin(theta_phis_in_pano[:, 0])
            # pointY = distances * np.cos(theta_phis_in_pano[:, 1]) * np.cos(theta_phis_in_pano[:, 0])
            # pointZ = distances * np.sin(theta_phis_in_pano[:, 1])
            #
            # points3D = self.rotate_x(-heading_of_pano).dot(self.rotate_z(-pitch_of_pano)).dot(points3D.T).T
            # points3D += np.array([cameraX, cameraY, cameraH])
            # distances = distances.reshape((distances.size, 1))
            # points3D = np.concatenate((points3D, distances), axis=1)

            pitch_of_pano = 0


            pointX = cameraX + distances * np.cos(theta_phis_in_pano[:, 1] + pitch_of_pano) * np.sin(theta_phis_in_pano[:, 0] + heading_of_pano)
            pointY = cameraY + distances * np.cos(theta_phis_in_pano[:, 1] + pitch_of_pano) * np.cos(theta_phis_in_pano[:, 0] + heading_of_pano)
            pointZ = cameraH + distances * np.sin(theta_phis_in_pano[:, 1] + pitch_of_pano)

            points3D = np.stack((pointX, pointY, pointZ, distances), axis=1)

            # print('final points3D shape : ', points3D[0:5], len(points3D))
            return points3D  # ， results
        except Exception as e:
            print("Error in getPointCloud2():", e)


    def seg_to_pointcloud(self, seg_list, saved_path, fov):
        try:
            if not isinstance(seg_list, list):
                seg_list = [seg_list]
                print("Converted the single file into a list.")
            # print(io.imread(seg_list[0]).shape)

            for idx, seg in enumerate(seg_list):

                try:

                    predict = io.imread(seg)
                    predict = np.array(predict)
                    h, w = predict.shape
                    # print("image w, h: ", w, h)
                    sidewalk_idx = np.argwhere(
                        (predict == 11) | (predict == 52))  # sidewalk and path classes in ADE20k.
                    sidewalk_idx = np.argwhere((predict == 11))  # sidewalk and path classes in ADE20k.
                    sidewalk_idx = [tuple(row) for row in sidewalk_idx]

                    class_code = [11] * len(sidewalk_idx)
                    class_code = np.array(class_code)
                    print(seg)

                    # plt_x = [row[1] for row in sidewalk_idx]
                    # plt_y = [row[0] for row in sidewalk_idx]
                    # plt.scatter(plt_x, plt_y)
                    # plt.show()
                    # print("len of sidewalks pixels: ", len(sidewalk_idx), seg)
                    basename = os.path.basename(seg)
                    params = basename[:-4].split('_')
                    # print("params:", params)
                    thumb_panoId = '_'.join(params[:(len(params) - 4)])
                    pano_lon = params[-4]
                    pano_lat = params[-3]
                    if len(thumb_panoId) < 16:
                        thumb_panoId, pano_lon, pano_lat = GPano.getPanoIDfrmLonlat(GPano(), pano_lon, pano_lat)
                        print("thumb_panoId: ", thumb_panoId)

                    # if len(params) > 5:
                    #     print("thumb_panoId:", thumb_panoId)
                    # pano_lon = params[-4]
                    # pano_lat = params[-3]
                    # pano_heading = params[-4]
                    # pano_pitch = params[-4]
                    # pano_H = params[-4]
                    thumb_heading = float(params[-1])
                    thumb_pitch = float(params[-2])
                    # print("thumb_heading:", thumb_heading)

                    results = []

                    if len(sidewalk_idx) > 1:
                        # get spherial coordinates
                        sidewalk_sph = self.castesian_to_shperical(sidewalk_idx, w, h, fov)
                        # sidewalk_sph = GPano.castesian_to_shperical_pp(GPano(), sidewalk_idx, w, h, fov)
                        # results = np.array(sidewalk_sph)
                        # print('sidewalk_sph[0]:', sidewalk_sph[0])

                        # obj_json = GSV_depthmap.getJsonDepthmapfrmLonlat(GPano(), lon, lat)
                        obj_json = GPano.getJsonfrmPanoID(GPano(), thumb_panoId, dm=1)
                        # print(obj_json)
                        pano_heading = obj_json["Projection"]['pano_yaw_deg']
                        pano_heading = float(pano_heading)
                        pano_pitch = obj_json["Projection"]['tilt_pitch_deg']
                        pano_pitch = float(pano_pitch)
                        # print('pano_heading:', pano_heading)
                        pano_lon = obj_json["Location"]['original_lng']
                        # print('pano_lon:', pano_lon)
                        pano_lat = obj_json["Location"]['original_lat']
                        # print('pano_lat:', pano_lat)
                        pano_H = obj_json["Location"]['elevation_wgs84_m']
                        # print('pano_H:', pano_H)
                        dm = self.getDepthmapfrmJson(obj_json)
                        # print(dm)
                        pointcloud = self.getPointCloud(sidewalk_sph, thumb_heading - pano_heading, thumb_pitch, dm,
                                                        pano_lon, pano_lat, pano_H, pano_heading, pano_pitch)
                        pointcloud = np.array(pointcloud)
                        print("pointcloud shape:", pointcloud.shape)

                        # self.saveDepthmapImage(dm, os.path.join(saved_path, basename.replace('.png', '.tif')))

                        class_code = class_code.reshape((len(class_code), 1))
                        pointcloud_class = np.concatenate((pointcloud, class_code), axis=1)

                        print("pointcloud_class shape:", pointcloud_class.shape)
                        print("pointcloud_class :", pointcloud_class[:2])

                        np_image, worldfile = self.pointCloud_to_image(pointcloud_class, resolution=0.1)
                        # print("np_image:", np_image[:5])

                        im = Image.fromarray(np_image)

                        new_file_name = os.path.join(saved_path, basename[:-4] + '.png')

                        print("new_file_name:", new_file_name)
                        im.save(new_file_name)
                        # colored = self.get_color_pallete(np_image, 'ade20k')
                        # colored.save(new_file_name)
                        worldfile_name = new_file_name.replace(".png", '.pgw')
                        # results_name = new_file_name.replace(".png", '.csv')
                        # print("worldfile:", worldfile_name)
                        with open(worldfile_name, 'w') as wf:
                            for line in worldfile:
                                # print(line)
                                wf.write(str(line) + '\n')
                        # results = np.concatenate((np.array(sidewalk_idx), results, np.array(sidewalk_sph)), axis=1)
                        # np.savetxt(results_name, results, delimiter=",")

                        # saved_path = r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024_pc'

                        # print('new_file_name: ', new_file_name)
                        # plt.imshow(predict)
                        # plt.show()
                        # plt_x = [row[0] for row in pointcloud]
                        # plt_y = [row[1] for row in pointcloud]
                        # plt.scatter(plt_x, plt_y)
                        # plt.show()
                        #
                        #
                        # with open(new_file_name, 'w') as f:
                        #     f.write('x,y,h,d\n')
                        #     f.write('\n'.join('%s,%s,%s,%s' % x for x in pointcloud))
                    else:
                        print("No point in image:", seg)

                except Exception as e:
                    print("Error in seg_to_pointcloud() for loop:", e, seg)
                    continue

        except Exception as e:
            print("Error in seg_to_pointcloud():", e, seg)

    def rotate_x(self,pitch):
        #picth is degree
        r_x = np.array([[1.0,0.0,0.0],
                        [0.0,math.cos(pitch),-1*math.sin(pitch)],
                        [0.0,math.sin(pitch),math.cos(pitch)]])
        return r_x

    def rotate_y(self,yaw):
        #
        r_y = np.array([[math.cos(yaw),0.0,math.sin(yaw)],
                        [0.0,1.0,0.0],
                        [-1*math.sin(yaw),0.0,math.cos(yaw)]])
        return r_y

    def rotate_z(self,roll):
        #
        r_z = np.array([[math.cos(roll),-1*math.sin(roll),0.0],
                        [math.sin(roll),math.cos(roll),0.0],
                        [0.0,0.0,1.0]])
        return r_z


    def castesian_to_shperical(self, theta0, phi0, tilt_pitch, tilt_yaw, fov_h, height, width):  # yaw: set the heading, pitch
        """
        Convert the row, col to the original spherical coordinates which can be used as the
         coordinates of the depthmap to look up distance.
        :param rowcols:
        :param theta0:
        :param phi0:
        :param tilt_pitch:
        :param tilt_yaw:
        :param fov_h:
        :param height:
        :param width:
        :return:
        """
        tilt_pitch =0
        m = self.rotate_y(phi0).dot(self.rotate_x(theta0 - tilt_pitch))
        print("m: ", m)

        print("theta0, phi0, tilt_pitch, tilt_yaw:", math.degrees(theta0), math.degrees(phi0), math.degrees(tilt_pitch), math.degrees(tilt_yaw) )

        # height = int(math.ceil(width * np.tan(fov_v / 2) / np.tan(fov_h / 2)))
        width = int(width)

        fov_v = atan((height * tan((fov_h / 2)) / width)) * 2

        print("height, width, fov_v, fov_h", height, width, fov_v, fov_h)
        print("height, width, fov_v, fov_h", height, width, math.degrees(fov_v), math.degrees(fov_h))

        DI = np.ones((int(height * width), 3), np.int)

        trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                          [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        DI[:, 0] = xx.reshape(height * width)
        DI[:, 1] = yy.reshape(height * width)

        v = np.ones((height * width, 3), np.float)

        v[:, :2] = np.dot(DI, trans.T)

        print("trans.T:", trans.T)

        v = np.dot(v, m.T)
        print("v: ", v)

        diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
        theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
        phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi#+ np.pi

        # theta = np.arctan2(v[:, 1], diag) + theta0
        # phi = np.arctan2(v[:, 0], v[:, 2]) -math.pi

        # plt_x = [math.degrees(x) for x in phi]
        # plt_y =  [math.degrees(x) for x in theta]
        # plt.scatter(plt_x, plt_y)
        # plt.title("castesian to spheric")
        # plt.show()

        print("len of diag, theta, phi", len(diag), len(theta), len(phi))

        theta = theta.reshape(height, width)
        phi =     phi.reshape(height, width)


        result = np.stack((theta, phi), axis=2)
        # print("result in castesian_to_shperical():", math.degrees(result[0][0]), math.degrees(result[0][1]))

        return result


    def castesian_to_shperical0(self, theta0, phi0, tilt_pitch, tilt_yaw, fov_h, height, width):  # yaw: set the heading, pitch
        """
        Convert the row, col to the original spherical coordinates which can be used as the
         coordinates of the depthmap to look up distance.
        :param rowcols:
        :param theta0:
        :param phi0:
        :param tilt_pitch:
        :param tilt_yaw:
        :param fov_h:
        :param height:
        :param width:
        :return:
        """
        tilt_pitch = 0
        m = self.rotate_y(phi0).dot(self.rotate_x(theta0 - tilt_pitch))
        # print("m: ", m)

        # print("theta0, phi0, tilt_pitch, tilt_yaw:", math.degrees(theta0), math.degrees(phi0), math.degrees(tilt_pitch), math.degrees(tilt_yaw) )
        #
        # height = int(math.ceil(width * np.tan(fov_v / 2) / np.tan(fov_h / 2)))
        width = int(width)

        fov_v = atan((height * tan((fov_h / 2)) / width)) * 2

        # print("height, width, fov_v, fov_h", height, width, fov_v, fov_h)
        # print("height, width, fov_v, fov_h", height, width, math.degrees(fov_v), math.degrees(fov_h))

        DI = np.ones((int(height * width), 3), np.int)

        trans = np.array([[2. * np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                          [0., -2. * np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        DI[:, 0] = xx.reshape(height * width)
        DI[:, 1] = yy.reshape(height * width)

        v = np.ones((height * width, 3), np.float)

        v[:, :2] = np.dot(DI, trans.T)

        # print("trans.T:", trans.T)

        v = np.dot(v, m.T)
        # print("v: ", v)

        diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)

        theta = np.arctan2(v[:, 1], diag) + theta0
        phi = np.arctan2(v[:, 0], v[:, 2]) # + phi0

        # plt_x = [math.degrees(x) for x in phi]
        # plt_y =  [math.degrees(x) for x in theta]
        # plt.scatter(plt_x, plt_y)
        # plt.title("castesian to spheric")
        # plt.show()

        # print("len of diag, theta, phi", len(diag), len(theta), len(phi))

        theta = theta.reshape(height, width)
        phi =     phi.reshape(height, width)


        result = np.stack((theta, phi), axis=2)
        # print("result in castesian_to_shperical0():", math.degrees(result[0][0][0]), math.degrees(result[1][0][1]))

        return result


    def pointCloud_to_image(self, pointcloud, resolution):
        try:
            minX = min(pointcloud[:, 0])
            maxY = max(pointcloud[:, 1])
            rangeX = max(pointcloud[:, 0]) - minX
            rangeY = maxY - min(pointcloud[:, 1])
            w = int(rangeX / resolution)
            h = int(rangeY / resolution)
            np_image = np.zeros((h, w), dtype=np.uint8)
            # print("np_image.shape: ", np_image.shape)
            print('rangeX, rangeY, w, h:', rangeX, rangeY, w, h)
            # pointcloud = pointcloud[:5]
            # print("pointcloud: ", pointcloud)
            # np_image[int((row[0] - minX) / resolution)][int((maxY - row[1]) / resolution)] = row[4]
            for point in pointcloud:
                # print("point: ", point)
                col = int((point[0] - minX) / resolution)
                row = int((maxY - point[1]) / resolution)
                # if point[4] == 11:
                #     print("col, row, row[4]: ", col, row, point[4])
                # print("col, row : ", col, row )
                # print("col, row, row[4]: ", col, row, point[4])
                if row == h:
                    row = h - 1
                if col == w:
                    col = w - 1
                np_image[row][col] = int(point[-1])
            worldfile = [resolution, 0, 0, -resolution, minX, maxY]

            return np_image, worldfile
        except Exception as e:
            print("Error in pointCloud_to_image():", e)

    def seg_to_landcover(self, seg_list, saved_path, fov=90):
        try:
            if not isinstance(seg_list, list):
                seg_list = [seg_list]
                print("Converted the single file into a list.")
            # print(io.imread(seg_list[0]).shape)

            for idx, seg in enumerate(seg_list):
                try:
                    predict = io.imread(seg)
                    predict = np.array(predict)
                    h, w = predict.shape
                    # print("image w, h: ", w, h)
                    # sidewalk_idx = np.argwhere((predict == 11) | (predict == 52))  # sidewalk and path classes in ADE20k.
                    # sidewalk_idx = [tuple(row) for row in sidewalk_idx]
                    print("seg files:", seg)

                    # plt_x = [row[1] for row in sidewalk_idx]
                    # plt_y = [row[0] for row in sidewalk_idx]
                    # plt.scatter(plt_x, plt_y)
                    # plt.show()
                    # print("len of sidewalks pixels: ", len(sidewalk_idx), seg)
                    basename = os.path.basename(seg)
                    params = basename[:-4].split('_')
                    # print("params:", params)
                    thumb_panoId = '_'.join(params[:(len(params) - 4)])
                    pano_lon = float(params[-4])
                    pano_lat = float(params[-3])
                    if len(thumb_panoId) < 16:
                        thumb_panoId, pano_lon, pano_lat = GPano.getPanoIDfrmLonlat(GPano(), pano_lon, pano_lat)
                        print("thumb_panoId: ", thumb_panoId)

                    # if len(params) > 5:
                    #     print("thumb_panoId:", thumb_panoId)

                    thumb_heading = float(params[-1])
                    thumb_pitch = float(params[-2])
                    # print("thumb_heading:", thumb_heading)

                    obj_json = GPano.getJsonfrmPanoID(GPano(), thumb_panoId, dm=1, saved_path=saved_path)
                    depthMapData = self.parse(obj_json['model']['depth_map'])
                    # print(dm_string)
                    header = self.parseHeader(depthMapData)
                    # print('header:',header)
                    dm_planes = self.parsePlanes(header, depthMapData)
                    # print('dm_planes[indices]:', dm_planes['indices'])
                    # print('len of dm_planes[indices]:', len(dm_planes['indices']))

                    dm = self.getDepthmapfrmJson(obj_json)
                    # print('dm[depthjMap]:', dm['depthMap'])
                    # print('dm[depthjMap] min, max:', min(dm['depthMap']), max(dm['depthMap']))

                    # self.saveDepthmapImage(dm, os.path.join(saved_path, basename.replace('.png', '.tif')))
                    # GPano.getPanoZoom0frmID(GPano(), thumb_panoId, saved_path)

                    url = GPano.getGSV_url_frm_lonlat(self, pano_lon, pano_lat, heading=thumb_heading)
                    print("Google street view URL:", url)

                    # sidewalk_idx = np.argwhere(predict > -1)  # all classes in ADE20k.
                    sidewalk_idx = np.argwhere(predict > -1)  #
                    sidewalk_idx = [tuple(row) for row in sidewalk_idx]
                    classes = np.where(predict > -1)
                    # print("classes len: ", len(classes))

                    print("len of sidewalk_idx:", len(sidewalk_idx))
                    # print("sidewalk_idx:", sidewalk_idx)

                    # get ground pixels
                    ground_pixels = []
                    # idm = dm['depthMap'].reshape(256, 512)

                    # for y in range(dm['height']):
                    #     for x in range(dm['width']):
                    #         planeIdx = dm_planes['indices'][y * dm['width'] + x]
                    #         plane = dm_planes['planes'][planeIdx]
                    #
                    #         #if x % 10 == 0 and y % 50 == 0:
                    #         norm = np.argmax(np.abs(plane['n']))
                    #         if norm == 2 and planeIdx > 0:
                    #             idm[y][x] = 100
                    #             print('plane planeIdx: ', planeIdx)
                    #             print('plane: ', plane)
                    #         else:
                    #             idm[y][x] = 10

                    # if  norm == 2 and plane['n'][2] > 0:
                    #     #print('plane planeIdx: ', planeIdx)
                    #     print('plane: ', plane)
                    #     print('max projection: ', norm)
                    #
                    #     theta, phi = GPano.castesian_to_shperical(GPano(), (x, y), w, h, fov)
                    #     url = GPano.getGSV_url_frm_lonlat(self, pano_lon, pano_lat, heading=(phi + thumb_heading), tilt=(theta + thumb_pitch + 90))
                    #     print("Google street view URL:", x, y, theta, phi, url)
                    #     #print("n2: ", sum(map(lambda x: x*x, plane['n'])))

                    # plt.imshow(idm)
                    # plt.show()

                    if len(sidewalk_idx) > 1:
                        # get spherial coordinates
                        # sidewalk_sph = self.castesian_to_shperical(sidewalk_idx, w, h, fov)
                        # print('sidewalk_sph[0]:', sidewalk_sph[0])

                        # obj_json = GSV_depthmap.getJsonDepthmapfrmLonlat(GPano(), lon, lat)

                        # print(obj_json)
                        pano_heading = obj_json["Projection"]['pano_yaw_deg']
                        pano_heading = float(pano_heading)
                        pano_pitch = obj_json["Projection"]['tilt_pitch_deg']
                        pano_pitch = float(pano_pitch)
                        # print('pano_heading:', pano_heading)
                        pano_lon = obj_json["Location"]['original_lng']
                        # print('pano_lon:', pano_lon)
                        pano_lat = obj_json["Location"]['original_lat']
                        # print('pano_lat:', pano_lat)
                        pano_H = obj_json["Location"]['elevation_wgs84_m']
                        # print('pano_H:', pano_H)
                        # dm = self.getDepthmapfrmJson(obj_json)
                        # print(dm)
                        pointcloud = self.getPointCloud(sidewalk_sph, thumb_heading - pano_heading, thumb_pitch, dm,
                                                        pano_lon, pano_lat, pano_H, pano_heading, pano_pitch)
                        # print("pointcloud: ", pointcloud[:3], len(pointcloud))
                        # saved_path = r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024_pc'
                        new_file_name = os.path.join(saved_path, basename[:-4] + '.png')
                        # print('new_file_name: ', new_file_name)
                        # plt.imshow(predict)
                        # plt.show()
                        # plt_x = [row[0] for row in pointcloud]
                        # plt_y = [row[1] for row in pointcloud]
                        # plt.scatter(plt_x, plt_y)
                        # plt.show()

                        distance_max = 20
                        distance_min = 2
                        pointcloud_clean = []
                        pointcloud_np = np.array(pointcloud)
                        all_classes = predict.reshape((predict.size, 1))
                        # print("all_classes:", all_classes[:3], all_classes.shape)

                        pointcloud_class = np.concatenate((pointcloud_np, all_classes), axis=1)
                        # print("pointcloud_class len:", len(pointcloud_class))
                        # print("pointcloud_class:", pointcloud_class[:3])
                        # print("classes:", classes[0][:3], classes[1][:3])
                        # print("pointcloud_np  :", pointcloud_np[:3])

                        pointcloud_clean = pointcloud_class[pointcloud_class[:, 3] > distance_min]
                        # print("pointcloud_clean len:", len(pointcloud_clean0))
                        pointcloud_clean = pointcloud_clean[pointcloud_clean[:, 3] < distance_max]
                        #
                        # print("pointcloud_clean:", pointcloud_clean[:5])
                        # print("pointcloud_clean len:", len(pointcloud_clean))

                        np_image, worldfile = self.pointCloud_to_image(pointcloud_clean, resolution=0.1)
                        #                         # print("np_image:", np_image[:5])

                        colored = self.get_color_pallete(np_image, 'ade20k')
                        colored.save(new_file_name)
                        # im.save(new_file_name)
                        # im = Image.fromarray(colored)
                        print("new_file_name:", new_file_name)
                        # im.save(new_file_name)
                        worldfile_name = new_file_name.replace(".png", '.pgw')
                        print("worldfile:", worldfile)
                        with open(worldfile_name, 'w') as wf:
                            for line in worldfile:
                                print(line)
                                wf.write(str(line) + '\n')
                        # plt.imshow(im)
                        # plt.show()


                    else:
                        print("No point in image:", seg)

                except Exception as e:
                    print("Error in seg_to_landcover() for loop:", e, seg)
                    continue

        except Exception as e:
            print("Error in seg_to_landcover():", e, seg)

    def seg_to_landcover2(self, seg_list, saved_path, fov=math.radians(90)):
        try:
            if not isinstance(seg_list, list):
                seg_list = [seg_list]
                print("Converted the single file into a list.")
            # print(io.imread(seg_list[0]).shape)

            for idx, seg in enumerate(seg_list):
                try:
                    predict = io.imread(seg)
                    predict = np.array(predict)
                    h, w = predict.shape
                    # print("image w, h: ", w, h)
                    # sidewalk_idx = np.argwhere((predict == 11) | (predict == 52))  # sidewalk and path classes in ADE20k.
                    # sidewalk_idx = [tuple(row) for row in sidewalk_idx]
                    print("seg files:", seg)

                    # plt_x = [row[1] for row in sidewalk_idx]
                    # plt_y = [row[0] for row in sidewalk_idx]
                    # plt.scatter(plt_x, plt_y)
                    # plt.show()
                    # print("len of sidewalks pixels: ", len(sidewalk_idx), seg)
                    basename = os.path.basename(seg)
                    params = basename[:-4].split('_')
                    print("params:", params)
                    thumb_panoId = '_'.join(params[:(len(params) - 4)])
                    pano_lon = float(params[-4])
                    pano_lat = float(params[-3])
                    if len(thumb_panoId) < 16:
                        print("thumb_panoId: ", thumb_panoId)
                        thumb_panoId, pano_lon, pano_lat = GPano.getPanoIDfrmLonlat(GPano(), pano_lon, pano_lat)


                    # if len(params) > 5:
                    #     print("thumb_panoId:", thumb_panoId)

                    thumb_heading = math.radians(float(params[-1]))

                    thumb_theta0 = math.radians(float(params[-2]))
                    # print("thumb_heading:", thumb_heading)
                    # print("thumb_panoId: ", thumb_panoId)
                    obj_json = GPano.getJsonfrmPanoID(GPano(), thumb_panoId, dm=1, saved_path=saved_path)
                    depthMapData = self.parse(obj_json['model']['depth_map'])
                    # print(dm_string)
                    header = self.parseHeader(depthMapData)
                    # print('header:',header)
                    dm_planes = self.parsePlanes(header, depthMapData)
                    # print('dm_planes[indices]:', dm_planes['indices'])
                    # print('len of dm_planes[indices]:', len(dm_planes['indices']))

                    dm = self.getDepthmapfrmJson(obj_json)

                    # print("dm:", dm)


                    # print('dm[depthjMap]:', dm['depthMap'])
                    # print('dm[depthjMap] min, max:', min(dm['depthMap']), max(dm['depthMap']))


                    url = GPano.getGSV_url_frm_lonlat(self, pano_lon, pano_lat, heading=thumb_heading)
                    print("Google street view URL:", url)

                    # sidewalk_idx = np.argwhere(predict > -1)  # all classes in ADE20k.
                    sidewalk_idx = np.argwhere(predict > -1)  #
                    # print('sidewalk_idx:', sidewalk_idx)
                    # sidewalk_idx = [tuple(row) for row in sidewalk_idx]
                    # print('sidewalk_idx:', sidewalk_idx)
                    classes = np.where(predict == -1)
                    # print("classes len: ", len(classes))

                    # print("len of sidewalk_idx:", len(sidewalk_idx))
                    # print("sidewalk_idx:", sidewalk_idx)


                    if len(sidewalk_idx) > 1:
                        # get spherial coordinates

                        pano_heading = obj_json["Projection"]['pano_yaw_deg']
                        pano_heading = math.radians(float(pano_heading))
                        pano_pitch = obj_json["Projection"]['tilt_pitch_deg']
                        pano_pitch = math.radians(float(pano_pitch))
                        pano_tilt_yaw = obj_json["Projection"]['tilt_yaw_deg']
                        pano_tilt_yaw = math.radians(float(pano_tilt_yaw))
                        # print('pano_heading:', pano_heading)
                        pano_lon = float(obj_json["Location"]['lng'])
                        # print('pano_lon:', pano_lon)
                        pano_lat = float(obj_json["Location"]['lat'])
                        # print('pano_lat:', pano_lat)
                        pano_H = obj_json["Location"]['elevation_wgs84_m']

                        thumb_phi0 = thumb_heading - pano_heading
                        # print('pano_H:', pano_H)
                        # dm = self.getDepthmapfrmJson(obj_json)
                        # print(dm)
                        fov_h = fov
                        fov_v = atan((h * tan((fov_h / 2)) / w)) * 2

                        # print('fov_h, fov_v:', fov_h, fov_v)

                        # print("len of sidewalk_idx:", len(sidewalk_idx))

                        sphs = self.castesian_to_shperical0(thumb_theta0, \
                                                                   thumb_phi0, pano_pitch, \
                                                                   pano_tilt_yaw, fov_h, h, w)
                        sidewalk_sph_phi   = sphs[sidewalk_idx[:, 0], sidewalk_idx[:, 1], 1]
                        sidewalk_sph_theta = sphs[sidewalk_idx[:, 0], sidewalk_idx[:, 1], 0]

                        # plt_x = [math.degrees(x) for x in sidewalk_sph_phi]
                        # plt_y = [math.degrees(x) for x in sidewalk_sph_theta]
                        # plt.scatter(plt_x, plt_y)
                        # plt.show()

                        sidewalk_sph = np.stack((sidewalk_sph_phi, sidewalk_sph_theta), axis=1)
                        # print('len of sidewalk_sph :', len(sidewalk_sph))
                        # print('sidewalk_sph[0]:', sidewalk_sph[0])

                        pointcloud = self.getPointCloud2(sidewalk_sph, thumb_heading, thumb_theta0, dm,
                                                        pano_lon, pano_lat, pano_H, pano_heading, pano_pitch)
                        # print("pointcloud: ", pointcloud[:3], len(pointcloud))
                        # saved_path = r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024_pc'
                        new_file_name = os.path.join(saved_path, basename[:-4] + '_landcover.png')
                        # print('new_file_name: ', new_file_name)
                        # plt.imshow(predict)
                        # plt.show()
                        # plt_x = [row[0] for row in pointcloud]
                        # plt_y = [row[1] for row in pointcloud]
                        # plt.scatter(plt_x, plt_y)
                        # plt.show()

                        distance_max = 20
                        distance_min = 0
                        pointcloud_clean = []
                        pointcloud_np = np.array(pointcloud)
                        # pointcloud_np = np.flip(pointcloud_np, axis=0)

                        # all_classes = np.flip(predict,axis=0).reshape((predict.size, 1))
                        all_classes = predict.reshape((predict.size, 1))
                        # print("all_classes:", all_classes[:3], all_classes.shape)

                        pointcloud_class = np.concatenate((pointcloud_np, all_classes), axis=1)

                        # print("pointcloud_class len:", len(pointcloud_class))
                        # print("pointcloud_class:", pointcloud_class[:3])
                        # print("classes:", classes[0][:3], classes[1][:3])
                        # print("pointcloud_np  :", pointcloud_np[:3])

                        pointcloud_clean = pointcloud_class[pointcloud_class[:, 3] > distance_min]
                        # print("pointcloud_clean len:", len(pointcloud_clean0))
                        pointcloud_clean = pointcloud_clean[pointcloud_clean[:, 3] < distance_max]
                        #
                        # print("pointcloud_clean:", pointcloud_clean[:5])
                        # print("pointcloud_clean len:", len(pointcloud_clean))

                        np_image, worldfile = self.pointCloud_to_image(pointcloud_clean, resolution=0.1)
                        #                         # print("np_image:", np_image[:5])

                        colored = self.get_color_pallete(np_image, 'ade20k')
                        colored.save(new_file_name)
                        # im.save(new_file_name)
                        # im = Image.fromarray(colored)
                        print("new_file_name:", new_file_name)
                        # im.save(new_file_name)
                        worldfile_name = new_file_name.replace(".png", '.pgw')
                        # print("worldfile:", worldfile)
                        with open(worldfile_name, 'w') as wf:
                            for line in worldfile:
                                # print(line)
                                wf.write(str(line) + '\n')
                        # plt.imshow(im)
                        # plt.show()

                        # for idx, point in enumerate(pointcloud):
                        #     if point[3] > distance_min and point[3] < distance_max:
                        #         pointcloud_clean.append((point[0], point[0], point[0], point[0], predict[classes[0][idx], classes[1][idx]]))
                        #
                        # print("pointcloud_clean:", pointcloud_clean)
                        #
                        # with open(new_file_name, 'w') as f:
                        #     f.write('x,y,h,d,c\n')
                        #     for idx, point in enumerate(pointcloud):
                        #         if point[3] > distance_min and point[3] < distance_max:
                        #             f.write('%s,%s,%s,%s,'% point)
                        #             f.write('%s\n' % predict[classes[0][idx], classes[1][idx]])

                    else:
                        print("No point in image:", seg)

                except Exception as e:
                    print("Error in seg_to_landcover2() for loop:", e, seg)
                    continue

        except Exception as e:
            print("Error in seg_to_landcover():", e, seg)

    def getDegreeOfTwoLonlat(self, latA, lonA, latB, lonB):
        """
        Args:
            point p1(latA, lonA)
            point p2(latB, lonB)
        Returns:
            bearing between the two GPS points,
            default: the basis of heading direction is north
            https://blog.csdn.net/zhuqiuhui/article/details/53180395
        """
        radLatA = math.radians(latA)
        radLonA = math.radians(lonA)
        radLatB = math.radians(latB)
        radLonB = math.radians(lonB)
        dLon = radLonB - radLonA
        y = math.sin(dLon) * cos(radLatB)
        x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
        brng = degrees(atan2(y, x))
        brng = (brng + 360) % 360
        return brng

    def get_color_pallete(self, npimg, dataset='ade20k'):
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
        vocpallete = self.getvocpallete(256)
        out_img.putpalette(vocpallete)
        return out_img

    def getvocpallete(self, num_cls):
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

    # def constructRawPointsCloud2(self, np_img, fov_h, theta0, phi0, w_thumb, h_thumb, depthmap, pano_yaw_deg, pano_tilt_pitch):
    #     # assert (self.c_panorama.pano_location_prop['panoId'] == self.c_depthmap.panoid)
    #
    #     dempth_image = np.array(depthmap['depthMap']).reshape(256, 512)
    #     raw_pointsCloud = []
    #
    #     fov_v = math.atan((h_thumb * math.tan((math.radians(fov_h) / 2)) / w_thumb)) * 2
    #     fov_v = math.degrees(fov_v)
    #
    #     depth_height, depth_width = dempth_image.shape
    #
    #     pano_width, pano_height = self.c_panorama.panorama.size
    #
    #     theta = theta0  # -self.c_panorama.pano_proj_prop['tilt_pitch_deg']
    #     phi = phi0 - pano_yaw_deg
    #
    #     # ndepthmap = np.array(self.c_depthmap.depthmap).reshape(depth_height, depth_width)
    #     #
    #
    #     np_img, pano_thetas, pano_phis = self.clip_pano2(theta, phi, 0, 0, math.radians(fov_h), math.radians(fov_v),
    #     #                                                  w_thumb, self.c_panorama.panorama)
    #     # img = Image.fromarray(nImage)
    #
    #     h_thumb, w_thumb = pano_phis.shape
    #     # ndepthmap_clip = self.clip_pano2(theta, phi, 0, 0, math.radians(fov_h), math.radians(fov_v),
    #     #                                  depth_width * fov_h / 360, ndepthmap.tolist())
    #     # ndepthmap_clip = np.array(ndepthmap_clip)
    #
    #     imgfile_name = r'D:\2019\njit learning\201909\streetView\StreetView\Pano_Depthmap\test1.png'
    #     # img.save(imgfile_name)
    #     colored = self.get_color_pallete(nImage, 'ade20k')
    #     colored.save(imgfile_name)
    #
    #
    #     grid_col = np.linspace(-math.pi, math.pi, depth_width)
    #     grid_row = np.linspace(math.pi, 0, depth_height)
    #
    #     # depthmap_data = np.array(self.c_depthmap.depthmap).reshape(self.c_depthmap.depthmap_height,
    #     #                                                            self.c_depthmap.depthmap_width)
    #     interp = interpolate.interp2d(grid_col, grid_row, dempth_image, kind='linear')
    #
    #     try:
    #         count = 0
    #         k = 0
    #         start_time = time.time()
    #         for y in range(h_thumb):
    #
    #             k = k + 1
    #             for x in range(w_thumb):
    #
    #                 lat1 = pano_thetas[y, x]
    #                 lat = lat1 - math.pi / 2
    #
    #                 lng1 = pano_phis[y, x] - math.pi
    #                 lng = lng1 - math.radians(phi)
    #
    #                 if lng > math.pi:
    #                     lng = lng - math.pi * 2.0
    #                 elif lng < -math.pi:
    #                     lng = lng + math.pi * 2.0
    #
    #                 depth = interp(lng1, lat1)[0]
    #                 r = math.cos(lat)
    #
    #                 color_value = img.getpixel((x, h_thumb - y - 1))
    #                 if depth > 0.0005 and depth < 20.0:  # and color_value ==6
    #
    #                     pnt_x = r * math.sin(lng) * depth
    #                     pnt_y = r * math.cos(lng) * depth
    #                     pnt_z = math.sin(lat) * depth
    #
    #                     if type(color_value) is tuple:
    #                         color_value = list(color_value)
    #                     else:
    #                         color_value = [color_value]
    #                     raw_pointsCloud.append([pnt_x, pnt_y, pnt_z] + list(color_value))
    #
    #                 if count % 100000 == 0:
    #                     t_100000 = time.time()
    #                     dt = t_100000 - start_time
    #
    #                     print("slapse:" + str(dt))
    #                 count = count + 1
    #     except Exception as e:
    #         print("Error in constructRawPointsCloud2():", e)
    #
    #     return raw_pointsCloud  # ,pointsColor


if __name__ == '__main__':
    print("Started to test...")
    gpano = GPano()

    #### Test for getPanoIDfrmLonlat()
    # print(gpano.getPanoIDfrmLonlat(-74.24756, 40.689524))  # Works well.

    # Using multi_processing to download panorama images from a list
    # list_lonlat = pd.read_csv(r'Morris_county\Morris_10m_points.csv')
    # print(sys.getfilesystemencoding())
    # print(sys.getdefaultencoding())
    # #list_lonlat = pd.read_csv(r'J:\Sidewalk\google_street_view\Qiong_peng\000_Residential_ready_regression_control_wait_street_quality2.csv', quoting=csv.QUOTE_ALL, engine="python", encoding='utf-8')
    # list_lonlat = pd.read_csv(r'D:\Code\StreetView\Essex\Essex_10m_points.csv')
    # #list_lonlat = list_lonlat[:]
    # list_lonlat = list_lonlat.fillna(0)
    # mp_lonlat = mp.Manager().list()
    # print(len(list_lonlat))
    # for idx, row in list_lonlat.iterrows():
    #     mp_lonlat.append([row['lon'], row['lat'], int(idx + 1), row['id'], row['CompassA']])
    #     #mp_lonlat.append([row['lon'], row['lat'], str(row['ID'])])
    #     #mp_lonlat.append([row['longitude'], row['latitude'], str(row['ID'])])
    #     #gpano.getPanoJPGfrmLonlat(row['lon'], row['lat'], saved_path='jpg')
    #     #print(idx)
    #     #gpano.getImage4DirectionfrmLonlat(row['lon'], row['lat'], saved_path=r'G:\My Drive\Sidewalk_extraction\Morris_jpg', road_compassA=row['CompassA'], prefix=int(row['id']))
    # #print(mp_lonlat)
    # print(len(mp_lonlat))
    # #gpano.getPanosfrmLonlats_mp(mp_lonlat, saved_path=r'G:\My Drive\Sidewalk_extraction\Morris_jpg', Process_cnt=1)
    # gpano.getImage4DirectionfrmLonlats_mp(mp_lonlat, saved_path=r'D:\Essex_jpg', Process_cnt=30)
    #
    # ------------------------------------

    '''
    Test for getJsonlink()
        Test Proj
    '''
    # list_lonlat = pd.read_csv(r'D:\Code\StreetView\Essex\Essex_10m_points.csv')
    # list_lonlat = list_lonlat[:100]
    # list_lonlat = list_lonlat.fillna(0)
    # print(len(list_lonlat))
    # mp_lonlat = mp.Manager().list()
    # for idx, row in list_lonlat.iterrows():
    #     mp_lonlat.append([row.lon, row.lat, str(int(row.id)), str(int(idx))])
    # print(len(mp_lonlat))
    # gpano.getJsonDepthmapsfrmLonlats_mp(mp_lonlat, saved_path=r'D:\Code\StreetView\Essex\t')

    # print("Started to download jpgs for trees.")
    # work_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\datasets\Waterloo'
    # saved_path = os.path.join(work_path, 'tree_jpg2')
    # list_lonlat = pd.read_csv(
    #     #os.path.join(work_path, r'Street_Tree_Inventory.csv'), quoting=csv.QUOTE_ALL, engine="python", encoding='utf-8')
    #     os.path.join(work_path, r'Street_Tree_Inventory.csv'))
    #
    # #m_point = int(len(list_lonlat)/2)
    # #num_sample = 2000
    # #list_lonlat = list_lonlat[(m_point - int(num_sample / 2)):(m_point + int(num_sample / 2))]
    #
    # list_lonlat = list_lonlat[11000:]
    # print("Got rows of :", len(list_lonlat))
    # #gsv_depthmap = GSV_depthmap()
    # #f = open(os.path.join(work_path, r'houses.csv'), 'w')
    # #f.writelines(r'row,id,ori_lon,ort_lat,lon,lat,heading' + '\n')
    # ori_lonlats_mp = mp.Manager().list()
    # print("Started to construct multi-processing list...")
    # for idx, row in list_lonlat.iterrows():
    #     ori_lon = float(row.X)
    #     ori_lat = float(row.Y)
    #     ID = str(int(idx))
    #     prefix = ID
    #     ori_lonlats_mp.append((ori_lon, ori_lat, prefix))
    #
    #     # print('lon/lat in csv:', ori_lon, ori_lat, '       ', str(ori_lat) + ',' + str(ori_lon))
    #     # panoid, lon, lat = gpano.getPanoIDfrmLonlat(ori_lon, ori_lat)
    #     # lon = float(lon)
    #     # lat = float(lat)
    #     # if panoid == 0:
    #     #     print("No PanoID return for row .", idx)
    #     #     continue
    #     # print('lon/lat in panorama:', lon, lat)
    #     # heading = gsv_depthmap.getDegreeOfTwoLonlat(lat, lon, ori_lat, ori_lon)
    #
    #     # print(idx, 'Heading angle between tree and panorama:', heading)
    #     #f.writelines(f"{ID},{ACCTID}{ori_lon},{ori_lat},{lon},{lat},{heading}" + '\n')
    #     # gpano.getImagefrmAngle(lon, lat, saved_path=os.path.join(work_path, 'house'), prefix=str(ID + "_" + ACCTID),
    #     #                        pitch=0, yaw=heading)
    # #f.close()
    # print("Got rows for multi-processing :", len(ori_lonlats_mp))
    # gpano.shootLonlats_mp(ori_lonlats_mp, Process_cnt=10, saved_path=saved_path, views=3)
    # print("Finished.")

    # ------------------------------------

    '''
    Test for computeDepthMap()
    '''

    parser = GSV_depthmap()
    s = "eJzt2gt0FNUdx_FsQtgEFJIQiAlvQglBILzktTOZ2YitloKKD6qAFjy2KD5ABTlqMxUsqICcVqqC1VoUqS34TkMhydjaorYCStXSKkeEihQoICpUFOzuJruZO4_dmdmZe2fY3_cc4RCS-L_z-c_sguaNy8rKzgrkZSGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBCbgsFgB9YzIFYFo_wdsAAZWbC5DliATCwYVPhjATKsYNC0f5s2baiNhagUDFrzb47aeMjVgkF7_liC06JgWv7YAX-n1rfjjx3wbVp-m_5YAV_mpH-bPGpjI4dy1h8L4Lcc9scC-Cyn_bEA_sq6vzF_zB8b4Ktc8McC-Cg3_LEAvkmH3wF_LIBfcskfC-CTHH37r_DHAvgj1_yxAL7IPX8sgB9y0R8L4IPc9McCeD9X_bEAns9dfyyA13PZHwvg8Rz11_LnFVI7SVZWO2UU_71-7jTybwd_67nuT28B4G8j-Gd2LvsXFtJbAPjbyLq_MT_8_RcFf1oLAH8bwd9fFRQUOPr9aPhTWoDT378glqPfEv4-qSCRo98W_v6owM_-dBYA_jaCvz_ytz-VBYC_jeDvsfprP1SeRd0_yVfA39W0_uV0_R393z_gbzX42w3-1rP-8g9_d9P4D4C_yeBvOfhTr7Q06W_D33Z-8Y-l-ABBrvYf4B9_LX_Cf6iTQxsFf8vBn3qp_FULAH_T-cq_dQHgr1v37t0tfgX8LUfJf6hp_-7KLB4G_paj5W96IL_5d9Yp-VeY9j839qPX_I35k_zxT9d_2LDmn3v06NH6wdPBP_kCkP4jR44s7T8y1pCWRkeL_2JI1L9Xr14K_16tpT0-e_8ePUj_jq2d3v6juUhRxNL-vZIV8Q-Hwwr_0YO02R2fhf-w-M8a_45kp4l_GVncv8xs5eWRHxT-0Y_1S6T8TOvjwz_NdP1LkhXDN_zdLjqVR_5R-Ot9ijZz48PffmdF0_XvZjtDTl3_Ir3OUhcZtahrJL0z0PIfWl1dPSqWrv-YWB70P6O1Tt9qqahoYLwuXXT9E5e-p8V0QWMp_I2cI3VqLqv5M7oqa9tWcZb48QJG_n2jFTdX0ru1qip7_pWVY6OFolVWVtacE68y5h_5UKUgCGr_wZr68Txf0jpQ_BwjonOS_n365ChqTxbIVpZzpiLlZYoVCASIX8eucGIXdP31bmA9LnVdOnXS0YwX91d9WCT2RGUejTwLmZF_RUVFbl9FxfFK7PkrX-IqiWL-Le9lVP6xz-5NVBVpRLRiorMj5eYS_sb47SPnJv1ziA04k7xmigun3YNOhLvunTtQ8bRIUSfDRN1vrSVvqyiZeyr_6GWraC43Wusm2PJXvcnppyjm3wxbpfKvimsXG9TMHq1Pczk66ehr_VUroLmAqquoswjKWl8p3MjI2wK7Kf9EFcpNsOOv8z63Kn5Hx_xbPFX-KdUT8HruSfR1_bUPgRRLYGIR9NfC9F4kdTZUN8Ge0j83d7je1YwsgQ1_o9tXk7E_YZ7ihk_ysp9I199gA1IugY1V0C0lcnr3ujX_WDpLYN1f9Q7CvD8pbsU9hb6hf47ey4ClLdDZBbPLQIncir_OElj11zFsKXFvJz6i8rfJnlo_mb_xQ8DyFljcCPe1tRn4a7ESS2DRn3jvoLsAirtc7W8d3QC_vfrcyfxzkj4E0toCz2Xi9lctgSX_FErq76_ytwOvp69z7hT-KR8CBlvguzWw5h_Ngn9Kp-bv2PprZ_xT65vwzzHzEPD_08DAP3JR0vW3I-eEvxl9c_5mHwJ-fh4Y-xttgCl-W3IO-JvDD5j1t7MCvtqE_Pwk_vorkNrfJn6stPxN61vwz7H0OmB2GbyyDfn5mg0g_HU2IIW_7l8bWci2vwX8gDV_-w8BK1HxVhf1z49cAWN_zQYk9Vd-4nCbq2DL35q-VX8aK0BBW1vcP1bCP1tdUn-jN4o6mVsJy_5W8QM2_F1fAbepdSP8m5dAq0-sgFXyVCXbADv6Jg9ux9_VFXCV2TCtv3EOoqfyzzHtbws_YNvfvRVwzzhZVvzdWQGjy2zG38ZzP559f_UKOLQDbgknz6K_GyuQZAGs6Fs8eFr-mh3IIH_HV8D4EifzTws_4IC_egXS3QHnbc1ky9_ZFbBx4dO0j-aEv2YHMsffwRVIx94mfsA5f8d2wEFUC6Xh79QK2MZP6-BO-mt2wMYSOORptfT8HdkBW_bp4Qec99cuQYb4p70D1vGdOLgr_tolML0FTpzJRs74p7cDVuwdO7h7_va2wLGDWcs5f_s7QJs-ltv-eluQbA2cPZ3pnPW3twS06WPR8TfcBM0quHBEM7ngb3kJktC7d3Da_qly76RJc8vfyha0XgQK7vHIQdm5x3P_xLq5ym9yD3JoeGuCfzQ6_im2gcnJ4R-Nhb8mJicnJmCtD3_qEROw1oc_9YgJWOvDn3rEBKz14U89YgLW-vCnHjEBa334U4-YgLU-_KlHTMBaH_7UIyZgrc_K392__jcbk6MTE7DWhz_1iAlY68OfesQErPXhTz1iAtb68KceMQFrffhTj5iAtT78qUdMwFof_tQjJmCtD3_qEROw1oc_9YgJWOvDn3rEBKz14U89YgLW-vCnHjEBa334U48cgTU_Q3825srYnJ0YgTU__KlHjMCaH_7UI0ZgzQ9_6hEjsOaHP_WIEVjzw596xAis-eFPPWIE1vxs_L3xx3_4w59BxAis-eFPPWIE1vzwpx4xAmt--FOPGIE1P_ypR4zAmh_-1CNGYM0Pf-oRI7Dmhz_1yBky1p8NORGTs3vsAcDkCsA_EfzZxeTs8Ie_okzkh39rmejvkbf_8Ic_i4gZ4M8uJoeHP_wVwZ9dTA4P_0z399ICMDm_R_jhD38mEUPAn1lMDh_IeH-vvPzDH_5MIoaAP7NYHD4aMQT8mcXi8NHIKTLSnw24KhaHj0VMAX9WsTh8LGKKTPP3zOMf_vBnEzEF_FnF4PDNkWNkoD8bb3UMDt8SMQb8GcXg8C0RY2SWv3ce__CHP6PIOTLOnw23JvqHT0TMAX820T98ImKOTPL30OPfM_4MF4D6wT3Ez9LfKw8A2sf20u0Pf_izi5gko_wZaWujfXgiYpJM8ffU7c_W3xsLQPfI3uJn7O-JBaB7Ym_xw5-yv8duf9b-XlgAmsf1Gj9zfw8sAMXDeo6fvT_7BaB3VO_xe8Cf-QZQO6f39L3hz3gBKB3Sgzd_tkf8mW4AnQN6U98z_gz_ayCNw3kUP9tD_gFWK-D6sbyLn-0t_2jkdL7397R9NFdPbztiRH_ye_UFX5Vbx3coYlY_-PuEvTWHz-9q5OQe4fcbuKr0L4CnUp0uTX6f25qKCos3Y33pvRNrCfqxvuIeKUvVBQ0jQy8dv5C7_YQk140qE4-83eepF-Z-wT24W5KXc2XigpsnbdxVvzm0-ZQkDy_sJRZ1v1n4a93vmy5ecTs3ZG_X8M79k0PrF3UNzftGkis79BVnnXw-9P6Yu7l_H5Pk54RuYtuCvsLaBePkg8d3b1pQf1CcW1Ar5F57lN8-OJc__itZvGPKkL6XjNvCn6qrlee8Wyqurb5OXpW_vemSPuWNt71WUBN8f0moYO_chuD_JPny1eXi2ZMWhW6pq-BrX5fknPMGidftnh5aOvylhkOHJXluTkfxH5-WC7vHjxM6S8P5rhUrxQtH1cqzO-1r_OxnbRqrpn5P3DbuDuHXt11affmCjvzkSavDnz8_T5i0eHH1-Y915jeWXCRW1LXhpq96gbvlkCQLh7qIQnZJaOs5O7hv9knyd0b1jcw_ivt0wmBueGSeV7qVicOO3yEf2XF-de23X2zo0P7l8DfTa7i3nsjj978nyVPaDxWv--JYqG7UWu6uI5J84vVssfc7S7gfHdjDf_fqWrlnzzXiwH43yBc98Jdq4doZDbMW7ArnPXkF94fupfz1b0ry2qfGiOUFU7mTg5bxA6-S5Jo3ponbP-rIHdh4Fv_xNknu-dYocdn6WfKkvZub_jOxd2P9_iPhpaHb5SfvW9j0xtbixp-ce1icemxE6Muc5dyrX0jyhyfyxPlPZ_GLd57kBr0iySsmni3-MHsPd2LgyYZ7XpPkZy6bL55x_ZpN0z7N4St2SnL_2lzxq9_dKry54ZdN78jLuKpFi8Rrjrbnp67b1rhpca28_IKnxaG_HTj2uaPjG_ZH9mdhzUcCt6eSG_O3eu6rA5HzF58SpoXncdMmzGl8dZEk3_f0LeJD22bLby1ZX716xIKGrEVbw48-_CB39d7_chv-KcmPbn1PuC1vjsBNXNc0uHAh9_gnPcWmLl0aTn18H79wpiSPLXpC-NOPV4cOX5zVUHVSklcW7xBKtrfj31-6heeX18qXzd0nHp5ZyAtlsxpXXC_JM2pfFIuXLOS4wjHcVZ9J8rEzy8Qb7_6aO1BXyq9_UZLnyJeLj7w0R9jz0Nrq-9_syn__vZlC0Q2z5WObn21a1K174_3zDof3DPhNw_qartWTv75LLlp-hfBo2XNjz7t3PP_1s5K8ZcXjQocr53NjpD3c0A8j86z5l9D5gUF8Xtk6vtfOWnnSlQPCo_94KdfuyM9Db38pyatu3iB-_uS73Auz2_HhP0vyzG4jxKcmb-Rq9q0MTYrsW1Hdw-K-D-7hz19X3zj-xlp5SoUY3lA_ketdsJM7GLkfHzv3A-Guv-fzB7aM5JetjJxv-kLx6ISXN065cwI_a70kZxVdKVx7sIo_OKy0cfUvJPnWxmvCcu-e_FUzahse3yzJ9bPXiHN2_GTTK5_Mb_xBxOPolEeqZxQIPBd4g__kzlo5f9op8YwdS7mbJr7LleyS5HvXfSnsWjVHvrXgmeqbLvxpw6G9H4n_Bz8_xLw"
    s = 'eJzt2nl4FPUdx_FskMASSEg2kIsYIBAMaiJgICS7mVlEDUf1QWupVqh3qQJaLFceYIqgRYvWthJFKq0glKsez2MVzSyjlKMiVGirxqMPrVoVtHg-FQrSzm6OvWaPmZ35zW_9fl5_8ORiMzvvz87uk2d7ejIyMjMcPTMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-JTlZ_dBgG2yguw-FLBBVgS7jwfYiuyPCdCi0R8TIES7f5wFnBGtB8PjBXPF6h-5AI3swf4qe44eUhZ7AB0LiFc-2B8LSFNx-mdlJW4f7I8FpCUT-2MAacjM_lhA-jG3PxaQbszujwGkF9P7YwFpxYL-WEAasaQ_BpA2rOmPAaQLi_pjAGkC_YlLfQDon87QnzaLngDsvluQJPSnDf1pQ3_a0J829KfAGfM76P_N53SiP11OJ_rT5XSiP11OJ_rT5XSiP11OJ_rT5XQm1z_uAIz3P5PdPQUNTvQnLLK-0ScA9E9LGvmtfAGA_pxBf9p46F_I8P5COPSnDf1pQ3_a0J829KcN_WlDf9rQnzb0pw39aUN_2tCfNsb9tQZwZiEGYB89_a15Awj62wn9aTPtCQD90xL604b-tKE_behPm2n9Df8BCP3thP60Jeifnx_6w9z0z_FL9a5DRoL--fnc9c_O6WLWKSAtTv_8fA76Z2cHPwpAf1PF7p9ve_9sLehvqpj981n176HdXzM--puNy_6x2qO_6fjrHy8--puNs_4J4qO_2XjqX5K4PvqbjJv-JSr0Z46L_iUd0J-5hP1DB4D-3wTDQz_hqn8yA0D_FJ1lvH_qbwBg2b-iIKC8vHyYqScwjQ1RhfeP-QbgtOnvcrmK_DIDiipUroJg-j59-obrmVhmtwAmSZgK9D9L_WB4VdXo9i_FugDw3_-881x-7dmL_OH9n4alN9BerZ_Z0T8C81qq3E55pR3690_h5kL7V7V_iYf-LteIEfr7dzzoO8OHlA-kj4yfVPue7VcRzf5M9tCrQ-9cLXl5ITvQv4Tq9v7V1RnD1QHU1NRkaPevrT3__K7-I0fWnR0w9tx6y_q7XA3tZ15H_8CjvqgoWL6gq3xk_OTad9ZPsn8nR_dwRnJr6q2KsYLgEPq3KxsYxyAto0b5_3Vr9T9X1dX_nIABqspKi_oXBbk6Je4fdsEPTW-sfbC-3v6OyAUkEre6xgg0Z5DXKXwI_i34DVYNDRgQl1Z_f-p2av_K9ltx-FnUv9gVpfP1XNQFoSt_1zU_tHxE_OTbh9bX31_3AoxsIMEMupbQP0xZuMGRtPo7YrGqvyozWlGoqP6dF_0Y7fWkD49vqL-BBegbQe8QCWbQNYX-msL24OClf2ABRRoqIi8NgeqB_upF34T2UfWN9Te2AOMj0NxB5AyCY9DegoOj_qo41UNe5kVf8_sYb69R32h_wwtIaQRaQ9CeQeggOvOb1z-pASToX6zxSiA8fFT5kPb602vXN94_pQnoGYHGBjSXEHcFgePlrH_YBArCRIUPyW8ofYz4qfVPbQHmjEBjDDHq89C_OEpBRHjN7p3xDaaPUz_F_ikvwOQVRC0ipD4P_ft1iVhAnO4h133T46fe35QF6FqBrhH07h1ypDz1D5VMeyse-ub0N28CVqwg9DB57R9zAqmmTxjfpP5mLkDfDnTU56G__4aTXECq5ZOrb1Z_8xegawfJ1GfdX2sAEb8jxgRSD59sfBP7WzYBXVOIU197Abb2j9LPjPB66pva3-oFJDsI_zdjH2Nq_YM_YKB_otOnYhvf7P6sJxBTgoPU1T_eDenon8S582Ma34L-XCwgmcNM1D-5XH5J9E_-xlIbgM74lvR32D6B5E927P56xemv85YYxreqv8PWCeg832aK7m_gRtjFt7C_w7YJGDjjJuvqb_D_M4tvbX97FmDwnHOFUXvdAzBwV5DfCFbxre_vh_z6MWnPqL-D2QTMO_-2Y9GeXX8HkwmYdOr5wKA90_5-qK-D5enZ9_dD_mSxyM--fwDyJ8Xa9Db290P-xKwsb3f_dsgfj9X1Lf4DYNKQXhuV_p3QPRy1_pGo9Y5EvT916E8b-tOG_rShP23oTxv604b-tKE_behPG_rThv60oT9t6E8b-tOG_rShP23oTxv602Z9f9vfAA5xoD9t6E8bZ_0xAMbQnzb0pw39aUN_2tCfNvSnDf1pQ3_a0J829KcN_WlDf9rQnzb0pw39aWPQHwPgGPrThv60oT9t6E8b-tOG_rShP23oTxv604b-tHHXHwNgikV_XAD4hf60oT9t6E8b-tOG_rTx1x8DYAn9aWPSH08A3EJ_2tCfNg77YwDssMmP_rxi1B9PAJxCf9q47I8BMIP-tLHqjycAPvHZHwNgBf1pY9YfTwBc4rQ_BsAI-tPGrj8GwCOG_TEADqE_bSz7YwD8YdofA-AN2_wYAG9Y99c5ACzAYsz7YwBcYd8fC-CJHf11LwATsIo9-TEBXtjX38AEsAHT2dofI7Cb3fGDdM8AQ0id3dFjMLAFzEE_uzMbYGwZ2Eg0u0tyzO40FrP79H4j2B1RW-r3KyPCox9WbNxfeqR16mlJOWdIiXjLO6_Ls5uekt99Q1LuP9lPdB4dstH52-W-2-ZJyoxfFosfDe4r3O69XBm9cYznrs8Oi7dm5jWsnXhMHvCepEz5IF_88oNLhM_k_kKvpq9bj6_u5r1hRbMwpvqyxvkH69yHTjwlXru6p2-e1CiXvS4p294oFPcvWiSMOVy54wdNbe6_bP6rePxoN8_pxUM98zZJSs1sr-gcXa7cNKNJ-dXda91b5lzv_ezvxfK9ayY37DslKW8cGyA-fKrOt-r4Uvnw05Lyi8WF4pdLuiu3XHmVsvKaPe7XXvyp96uCbM-dE6Z47lsuKXOLp4u_eeeVhr7Nl_iu3iop6-oaxE3fa5E_WTPVPeETSXlvfrm4atEIYcKwOuXx7ed5Zi50eytrjsiz2j6Qa16WlBfHDhSXrHT7Nu3-nbx7m6SM31kgPlnZ1nrxt951T_2XpNQO7C5OPrlPbmv7rrvifUnZf6hYrDp2i_Dt7Xsa55aOaLjwxAHxw0UH5XWNB-WWVyXlhy3l4sgy-fmfXzCwwfc_Sfn8wX7iUFep8tzaScoL0ze4b236iTd37G3unetrPFtlScl87CrxVN0VcuHgLN9k__n7zh-FpZ88KRde0Vq_4VNJ-TgnV3zz_Xrh7A-rhekjpspvPrK-ceSkHr5lu3r4ardLyhOXjRBfernZc2rBPE9OjqQsyM7xHl9-QM7u9rm868_q-a59UHh6zHzZ9acMefl_JGXPtl5imXCv7_X_XlR_7Hq1_817hJcevVF4a8o7jQ-ffmDM8F5t4-onLJAvXLpFvvioen8WHBGePVnuWftQX0_ZBknZPmOV-G6zKFTNHaRMe378jibnLO9DT8wRtg5dt-PshzI9C6sc4-S9JUrzg5OEywtG-z4te9S78pn5wjODljb--6N-7p9Ne3zcjopq99a6ia2nTkrK1lX3iB9nXu_-8ZI5rReckJRhpZPEhl9_2Tr7wlJf3iuSUv-PXULD4y75kRVCQ-7Xaq9z3ha_OnDAs_lKr-dvNy9RbvzisHd3n-t8dxxZ03rNGkn5onalcOenpfKM4YN8F--TlJK8ZeJVd7_tPmN6jXuJuueLHjgkNmU-4PnRyWc9X0xcIuTedo-4tlxw_3NTD3mS-vsnT_uDOOeOFs_i5466D4ySlFdH3uCtPHx8x6iZ3xeyGsf65m4eMO7SvfW-uRUbW089LynLnsgQW7askH_fXO6r3q32fOt-4bXbD7k_yvG5y9THW8-mneJq6Wr3tQeHyC-o53_XrAniTY9d6l5YdZ1vZ4uknP_wenF88W75q5mDfMXq7d23d7Twf54XJuY'
    s = 'eJzt23twVOUdxvENyyWsuQCBgAQMAUISJIkRIcRzNnsWtO000k7RVsRxWpnUglq8YOsF9DgoWBGl2inDZRputmKlVhGtuJucyKglIoKXdixKFVNEp4JKFSlQ7NlbspvdbM7unvf83pP3-f7DkAB5z_t5SEgmZM9wOPo4srIdCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEkCj16S7qgyGGdauOKfTuUoPHDHpPmchjBfbOLHtswH6ZbY8N2CdW9tiADWKNr9evH_VDosRZgB_0xwI4zBr8sD8GwFmW6Yf9sQC-st4fA-ApAn8sgKco_DEAfiLxxwK4icgfA-AkKn8MgI-o-DEAPqLzxwJ4iNIfA6CP1B8DII_WHwOgjtgfAyCO2h8DoI3cHwMgjd4fA6AM_mLHgT8GQBgP_hgAXVz4YwBk8eGPAVAFf7HjxB8DIIoXf14H0Dcm6tOYHzf-HA2gr7Goj2lK8I_OoHwvmgE__sQDSIe-F4yAI3_aAWTmb9sRwD-cCf523ABP_qQDMMnfbhvgyn8U3T0MNtHfThPgib8X-dtmAvAPZb6_PSYA_1BM_G0wAfiHYuXP-wTgH4qhP9cLgH-wwUz9OZ4A_IMx9-d1AfAPZoE_nwuAfzBL_HlcAPyDwR_-8BfWfzD84Q9_-As3APgHEtM_JydnIPwDCeefE7APBP9AYvlH7OEfSRz_nJycaH74BxosjH8O_BMEf6H9XQL4Dxo0KPAD3_40V-MSxD8Q_OODv9D-LviL7O-CP_zhL6q_C_7c-BN8-ueCP_zhbzv_oUNNuRUX_O3qb8oA4G9ffxMG4IK_jf0zHoCr9_sPC9U7_TNdgDj-HQOAf2cu-MPf_v4jQiV-pY39i4qSPLQR__z8_GSvdsGfc_8kA-jJPz9UkmtzwZ97_24GUFpamtQ_v6Mk1wZ_G_h7R-t1fabx3fgPD-RwTMg34u8S038Qkb8R_gT-o-P8x-sl858wAf5R2ct_ZLDw0Zn6u-DPuX8R_I0lhn-JnsdhxD8f_p3Z0T8nePIo_5JwtSUl8I8L_uHgH58t_SsDdfpXwj9Zhvzr6-EfXS_yd0T5n6dX3ZEkSRF_t9tG_oG6-lfo1cqynJJ_93fW-_zPjem8cBH_6mqb-Sud_gH6ippwIf6If0VUvdX_rHMSdHaSJkXSdxDSDyzCZv4Vsf41FwSbMiXWvyo6u_qfZbyuMxgbLNkYOtZgM_9pQf8yvWnTpuiVxRb2j3lZvH9ZWXl5-eRAeVFNnTq1P5X_kNhSoE82hbExJd6BzfzLI9XV1QVw8_Imd7xI9wz7R7Pmxft3gPfvGif-JuwgegtjEzYumL38p5d3KS-2BPwJ_OPYx0TizN-kHUTW0KnemR3847y6TcePe1m3_mM6i2yFyr-nAWS6g67q4XJzc3uXf4KGx_vHmoeaGIjs_b9B_9SGkJtCXPt3eSefd-GFF2bkH-c-MfKLyfyzYstoCqm429I_qijoBO_OOz7858X5x7D3580_ebED6HhxOu6BBuhx7Z_U13jd__uvmxlY6D8gWEoriC1N95g49U8mZQi-8--6Qf9wlvt3xk4_Tj1BtvdP_N49MgGj_v3p_FMZgynkXbKLfzoZ9i_gxD_pLjKEFtDfqL6eRf7muZmV8P4FBRb5FxdTWydKcP-QfoGTrb9TrzgcNXiXhPaP6Osx8nd2ibcVFHLkn9mXfzPBZ-HfVT7RCqhnUFgorH-svqn-yeR52kFhIWf-50eyXF8vc_eOz9OND4BwCUH9rCye_JNdEFv8DDaQ8Cs1aSwg8RwY7aEwrC-gf_f6CTOizWIBcZmtHz6nWP4FKerrZYczLM9oAkz0xfJPHd-Zib-5C2CBL5J_WvjOzPzNnAATfa78kz4-Db4zY3_TJsACnyv_5I9PpG-GvzkLYIHPkX9Pj0-D7zTH34wJsMDnx7_HxyexD2SSf8YTYIHPjX_Pj0-D7zTTP7MNsMDnxd_A49PgBzLVP_0JsMDnxN_I49PYBzLbP80NMLDnw9_Y49PYBzL3A0DaG2Bgz4W_wccnoQ_Gyj_FDTCwt84_Y_7k_qzog7H0D2bsGObTB6P1N34DFPKhmPsbuwLz6YOR-qdyBQnhTSJOmjX-PV6B2fDhKP1TuoQIOhPjZFnpn-QO4txNektk_qndg_nfUGE0En_rovJP9ZzwZxORf8rnhD-bSPzTOCf82UThn8454c8m6_3TOyf82WS1f7rnhD-bLPZP-5zwZ5Ol_hmcE_6sssw_o1PCn2FW-Gd4RPizjbV_pueDP_MY-md-OPhbEgN8cw4Gf8viDz8L_hbHE30w-FsfJ_TB4E8WsXwo-NNnvXpnZP5M_gMISjVyfwyANPiLHfzFDv5iB3-xg7_YwV_s4C928Bc7-Isd_MUO_mIHf7GDv9jBX-zo_PENADxE748BUAZ_sYO_2MFf7OAvdvAXO_iLHfzFDv5iB3-xI_THF4A5iAN_DIAw-Isd_MUO_mIHf7GDv9jBX-wo_TEA-uAvdvAXO_iLHfzFjtQfAyAP_mIHf7GDv9jR-mMA1MFf7OAvdsT-GABx8Bc7-IsdtT8GQBs1P_xpo-bHAGij1ndiAKRR4zvhTxo1fiAMgC5q-2AYAFnU9KEwAKqo5SNhATRRu3eEAZBEzR4VFkAQNXpsmIDVUYvHhQlYGjV3wrABy6Km7j6MwJKomXsOQ2AZtW6qZWMO5kYNyiBsJJWotciivni-otawMOqr5idHlw6_XNWUn1vua_tG1QonjlRGjr_Dc2LZ7Japh5vl2Y5q5YOXJkh3zB0trdFf_9PqkcqwA6qW_flW2XV0t_9Hw7K8nq2qZ_t3JrsPnHldfmtJrnKOb5dU-fUj0qgzqva31WcrjbvW-Lbce4e_6WtVm3TnUCXwNj-9_WN53SFV2yiPVC75SPVcu6BBPn_9Hvm8q6_03jxf1VY-dnez17fX_7NlqvfWJVvkI7-vkNcfVbXS2ZOU5_-x2LPponNaGg_tlB9u_a534cYbfVfccoO88oSqtbkrlWU1k2XfEIc89KSqvfnXIcqBixZpyqLp9f95X_Pv_G2rss6xwv2XRtUzYuAZac6_F3jHPfNpS-uZuZ4vZu72vTbgUs_vCp-uax9-if8T_fcvGpOrrNhSJDW-n-u-6x1VW_zEIGX9jxs928Ydbrl45Qb5ptKDyjd7i1v2vbfYc-_RAdL86_d7917-2I73ncubndep2ufVA5Qvqpc1u29StbtbTvm237XSW1lR4C56W_X84KV2qc8vb52-uvQW90X9iqSGB1TttrKLlbVTX6t_sfVaban7Yck9_rj3_gOqZ8mUP_o3Hdst7zw5yttwUPXkVq6Ub3l3j9wona0Mv6LKc-x6Sdv27Pfk7Jlzpr88cINvXvaHLxTrXtv6OJQdrUt9V52Y5l7tV7XlD01R2lfXyRtWVTev0FRN-_4sRbvyUn_tf1Vt6P_2-Xa--60Zs-YNdz_0zwkvTH5L1VpfLlWe-HmbvL2yuL5h72KtftnzysZ1N2q7rnqyZcYf3vBvLDrmffqyhdr6-5pa_tXwpv9UVZv3pnZVG1hwl7_8pT3-8dIwb5Xvftl1cql88oh-H21fecbccLnsdSo7fqjf747WCmXJMw_IV9_ez9_wpapVPb5IaZu-Qjq4YZG76R5VWzfwNc9H2Zvdo1_fLNVUq9qj9bcpxz9R3F_ULJFefE7V5lwnKe1nnvPX6ttYv-0d3wNf1sxY9dlm-ear65qf_bOq7X3qXeXINQvdC-pulZ-7V9_3DXOVR6SS-pIjQ-T3di_WJs16W2naXyT__YI1vlmnVc2_tVGpOa5JhxpP-Fs-ULWC5kqltLZCXvqx033PflV7cd6fPMeWt8ilxy7zHdWf5_Fff1up_WaHfHr9q9J-_ef995Yqvr6bpEdfXeq-YKGqfdZU5tl-6k7tcPmp5vYH3_Cf-1Srd957z8ofrclrHrVL1a7ru1WZ37ZGfmffgOZ5-1Stvni5Unv_RmnElrnuV9eq2m_mr_bM2rFKnumdLJ_-XNVU50HPI1vXyqefmt1832pVO9JY6W3IGetuf6NOvl3_8_a9MlGpmFlWP-IXr8gfL1qsffX2WO-yZ_p5fjJzjja-KUs-PuH49LnjnpTPlNztn_SJqk07uEl58EPVU3p8qVz1-h750KcfKGuvuVlbWfJoS-GIN_0Lhv1K-T_n7FK8'
    # decode string + decompress zip
    s = 'eJzt2n1wFPUdx_HkIEAgJCSBEAgYQ-QhjwZMAya7d3ukUQd0bBWpYKsVGhU6CtRpRCWsIFUzBTQoRSzWhxGUalsfsAPJhqVqFbWigGgFNR10QHxAKlaRB-09Jbm7XC6395DPHr_P-x_gJuz97vv67iXc0O_HCQmWhMR-CYwxxhhjjDHGGGOMMcYYY4wxxhhjjDHGGGOMMcYYY4wxxhhjjDHGGGOMMcYYY4wxxpgo9Qol9CFZ1AvJ3as-6AOz6GQU3sPvCH1yFmHh0bf7cwPiuAjsO_y5APFZhPgd_tyAuCtyex9_LkA8FR18H39uQLwUNXw_fy5APBRNfV_-PsnJyehXx4IWVfxA_twAExdt_YD-3ADTRn-xi7F_Mv3NHf3Fjv5iR3-xo7_QxZi_3b93797ol8oC1EP-vd2hXy3zr2f9uQJmq8f9uQKmKsb-yYH8uQLmCeTPDTBJMH-ugCmiv9jRX-zoL3b0Fzv6ix39xS62_AE__qO_iaK_2NFf7OgvdrH1549_Zo_-Ykd_saO_2NFf7OgvdvQXO_qLHf3Fjv5iR3-x6xH_gPz0N0P0F7uY8tPf9NFf7Ogvdjh_9CtnzmLqz3_-mT76ix39xY7-Ykd_saO_2NFf7OgvdvQXO_qLHf3Fjv5iR3-x6wn_gPz0N0X0Fzv6ix39xY7-YhdL_p7471-pfbsvSk91Whbn_qmh-HMRui6u_VNT6R9h8eyfSv-Ii6V_bD_-S02lf-TFq39qKv2jEf3jq6FRvh7946qh9HdH_-hkFv9MZwbOLab_0NPHP8WnTKH8C8L8e0Pp35aI_qPo3144_HHuP6rNv7g8ekcxh38m_buvzb-Y_iL7F9M_QVB_5wJMor8z-kcnM_kbObdI_hUVFc5fvPwn2aN1lB7wD8hP_24rba-iorSsrMztP7TM3eTKyvHuJOns9iqNHiVe_IuKinz-HBf-uW2NzO4oKyurICtrtFcjR3p-c66jke6sVVXj_PL4j5vgbkwXGTuimfzHds51RsWBX-jK0p5Z_AcO7NdR24M5nRs8eHD7NuQWODbCUXa271p49SNXnu3w5PG3lbjLC1bI5zeXf6Kzic6K2nI80C5_jqOzXJ3pKib-6enp_R2FQu9dP7-qh3c0xJH3MhR4NsJVx1q0S_tuhiPHlznE3f5O-ywjBTh6RkZGmjsz-Se25XevO_NS926QIf8BrvyHMcyRxZLuLikpqX9HXgdIDNTATvmvgafhPhW4NsJd5_eJQHm-gzj9vdZihIEyXKX5h_IP9PGvJVCeO95pbaQB3TTMk_MpOui71O_C3-gaeCrw2wffd4rAuffA4d_Fhvg2ZEiGV53U8f7eo01JsVg6-5_VlvtGj4K5v3sn-671g_sbXYOCgI9mBK1tD4YG2I1AXx_EPC2t40ljyB-yf7CiQO7t7n1pP_pg-KH5h7wHgf2DF3w7jJD7ZHZ_3x0ImbuLu71L-u70Dfl3uwhR8Q8Du2f9g378Z8TfuQFhqAe-VCD67vXD8w_9HQFRfn5-YiLC35i9q4jIg8uHpB-hv9lWwSnvrof9k5LO8C0c_xCsA9IHkg8NP4r-3W1DjNchv4O-B_27mLy7M_wzIhuRfOj6sfIPvWjQB7hsD_gHxQ-1MOGDyRvRj2__gPLtxdLf9QShQETPPyR3g_jx6u__bh-06Pp3vn7oMGH5G728If148zfi3qmI_YNfPqw9iJZ6ePjx4m_ofg8pw_5GLm4Az0_cMHiE-mb2jz5693mZR-2aof3EFnHh4JvIH4BthqKvbwTfBP6Ch8WnP7po4oehT39wWHz6o8Pi0x9dFPDDt6c_PCw-_dFFYh85Pv3R4W58-psipD398SHt6Y8PSE9_EwSkp78J6o4-dvL0N0MwePqbIl_2HkLnApgmgDj9TRSan_7Y0Pz0x4bmpz82ND_9saH56Y8NzU9_bGh--mND89MfG5qf_tjQ_PTHhuanPzY0P_2xofnpjw3NT39saH76Y0Pz0x8bmp_-2ND89MeG5qc_NjQ__bGh-emPDc1Pf2xofvpjQ_NzAbCh9emPDa1Pf2xoffpjQ-vTHxtan_7Y0Pr0x4bWpz82tD79saH16Y8NrU9_bGh9-mND69MfG1qf_tjQ-vTHhtanPza0voULAA2Nb6E_NDS-hf7Q0PgW-kND41voDw2Nb6E_NDS-hf7Q0PgW-kND41voDw2Nb6E_NDS-hf7Q0PjO0DMQObS9M_QMRA5t7ww9A5FD2ztDz0Dk0PbO0DMQObS9M_QMRA5t7ww9A5FD27tCD0Hg0PSu0EMQODS9K_QQBA5N7wo9BIFD07tCD0Hg0PTu0FMQN7S8O_QUxA0t7w49BXFDy7tDT0Hc0PKe0GMQNjS8J_QYhA0N7wk9BmFDw7eFnoOood3bQs9B1NDubaHnIGpo9_bQgxA0NHt76EEIGpq9I_QkxAyt7hV6FEKGRvcKPQohQ6N7h56FiKHNfUIPQ8DQ5L6hpyFeaHH_0PMQLbR3p9ADESw0d4DQIxEqNHbg0FMRJ7R0l6EHI0po52ChZyNCaONuQw_oNA_NG3roSZ2eoVXDDj240yW0YyxDzzY-QiuhQs_dZKE5eiz0oE1Ugl9HFqxq2tP6SnPL96q-pSlbaRi4uKo4_2DTdT-o-nPfZyrLF9TZvn6_Yat92n4pJb_EXrSnznYi-c6tv_7qI6lo2QK78v4rzaWrXm9edkrVr5-do3x6wQJb2nMbras-K5HnnzqqLG_trV_UOFO_8cSYlgMV4ydfKa3Wfrt6urTkS1XPe220UrhLte3YkyMV5w2Wj2wttn90TkLV3VOXt3xeq-rz9p2trL7pJemxRRO06kOqvj07X3lgwy-lTR-r-uh-b2n75n9hzy9do134umQtbLxVnzDicWXq26p-Zk5zc2FzZssLfWbYP53zsPa_4cutt5fU6cfuPak89PEyaVL1xKrcb1V9Zvog5eIFCdq9aWvkuqmqfsuWV227zhol3b_UUrXxpKpfs3eI8tmMQfrBMZfq65cWtsw4tsJ-cMXj2vEZ-6p2HVH1mpJRSlJuvjZ1zuXNtY6vL60doCRPeFnbNG2xfMkNqv78mZuUxj2q7c7tf6u012TKHx-YqDy84IQ2pXGkvOZpVV-5UFUm2vtUln33qFxQpuofNt1hu_qF8dq4J2qsa4tu1Y9tsdjX3rRRe2fQG_LyJxbr_WubldtSjmslGxKtO3fW63v3_6Ds_axVa9pxX9U7B1T9aMMIZfPBcmnWealbf3pbvT634R4lt7lVm1X3B3lWharf_erjyrfXTpdnPjO6Zf58x3zrb1DWX3-ddu3BPOvJbYv0nClz7fNuVm0lSy-Ta2_MlqU3HlQaFuZsWXrRqK2rXlykf7_NZvvJ3qIq7dEPm6sd-6LPyVWO939T2zF3s9zyn8X6_ZkJ9tq_TpHHnLfRurZmgX7BlEGTayacksZuf1I7b5eqr269TflX0_Tmv-TdJa39RtU_GTdDySm-XP_9n_vpyUfLWyrHP2W76ndW-RdHD2tjH1T1kylLlW2758mTzi9seWu6qk996BHlj-dPtx4pu8XWdEsf-esNg6rffPmfmj5xbVXp547rHRyhPFJcJ21ftkb61ReqnrS7Trksra_8Tcl98hLHfHfWn2Gf-MAb0oHrV8qPXKjqa-vL7bsX9pV33PSDfGpfvf7dgEZ763t3bWk4sUT-8FZV71XS255T_p62-dAKKbdV1V98erPtqdkL9ZlZddZzn_9UO370YvvJ0rs05e2Eqp2O15PRkK5ctfJG_ecjH7Tu7_uuVtsyavIdc95tPpVUq5UeU_WXqscq64bP19eNfca66h-vaVcU9pqcUfa1tvd4Y5W0X9U3lOcoc6-4R6qrmSZd6tivkxfXKDfPmKV9sGy19PZhVS9vuFpZV_KWdPh4pdTo8F49QFaeHDGhpddXj2l_36Tqj45fafv3J89K5z57u7zpOtWWeuhKZcmfdmnz-vxM_u_dqp6tb1OGqZe2fLm7QBq4XtWPpB6yrj-wyNb62DDrvVfkyXlH6idP2nGN1FTdKn_wm8W2lPTZ9uJpVu3zS1ZYZyfX2b47vE75P5RJCQo'
    s = 'eJzt2AlwFvUdxvHc70tCMMRAQjgk3IkpRUsg5ODdN0pUHESGoRapLTNApVUKrS1HSbKDrUYYjtJ6QBFECKIcsYgwxHffrO2oVSoI9aLA0EIpkVIVBOvRAn3fNwdvkvfY3Xd3f__d__OdTGYyyYTN7_PkGJylcXEJcfHOOIQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhNpKaBf10yCjS1AV9dMivVLnjh3YJkeg2PgxA0vmuJY-_hiBZXK0T0d_bID1HJ3T2R8bYLUQ9gb5YwLMFQbfMH-rTiBeSdQPqbbw-Eb6W2kCitgtuoOI-sb6W2AC2uQtM4Mo-Mb7M7wAHegZH0F0feP52VyArvaMbkCBvjn-jE3AEHvmNqBI3zR_VhZgqD1DG1Cob6I_AwNwmqLPwAQU65vpT7wApy_z_EkXoILfVH_CBTidZvuTTUCNvtn-NANwtmayP8kE1PGb7U-wAKeT0N_sBajUJ_A3eQBOJ7G_qQtQzU_gb-YCnE4G_M2bgHp-En-zBuDsGJm_KQvQoE_kb8oAOumT-hs_AE38RP7GDyCEPq2_0QvQxk_lb_QAQvIT-xu6AI38ZP6GLiC0Pr2_cQvQyk_ob9wAwvEz4G_QADTzU_obNICw-kz4G7IA7fyk_oYMIAI_G_76DyAGflp__QcQSZ8Vf70HEAu_zfwj87Pir-8AYuIn9td5AFH4mfHXdQCW9td1ANH42fHXcQGx8ZP76ziAqPws-es1AKvz6zaA6Pps-eszgBj5WfDXZwBK-Nny12MACfBvThE_Y_46DMAW_joMQBk_a_4xDyDBHv7dTeJnzj_GASTAXxU__G3pr5ifPf-YBuA_HfxV8NtrAIHT2cM_pgFY21_7AAKng78afhb9tQ6g-XTc-6viZ9Jf2wBaTmcTf4c5_PC3mb9Kfjb9tQyg9XR28dc4AHv4axhA6-ls469pAGr5WfVXPYC203Htr5rfLv7XTmcff_UDUM_PrL_KAVw7Hfzt4a9qAEGns5G_2gFo4Ie_ffy18DPsr2YAQaezk7-qAWjiZ9lfxQCCTgd_-NvGX8UAtPEz7Z8Of_jz7a98AHb0VzyAoNNx6p9mS3-lAwg6nc38lQ0gLQ3-LfHnnxbInv4KBxB0Orv5RxtAWhr84c-pf9eu9vePMgCb-_fsGeFrd3RtKeh08Ie_vfwjDiCDY38H_G3tHw__aAPI4NjfAX_4c-EfdgAZ8Ic_p_4OjvzDDYAL_6yskP8LBH8_v_39s0L7O-DPj3-IAfDlH3IAGfCHPx_-nQbg4Mw_Oww__OHPg396KH6e_G_oyO-EP1f-ffv25dk_PQQ_V_7Z8OfdP_iPACdv_rmm-JORtyukf3_4d-Dn2N8Jf_jz5Z8L_2B--MMf_vDnx98Jf_gH83PnP4Rv_1z4wx_-8Ic__Dn2H_4N-MMf_vCHP_w58x81Cv68-uc3V1RUBH_u_Ivz8wsKEhMTS25siT__MXFxI30vI7nzL2guL69PyIJOZ1__YW31C8SJf15QAwf6XnXrXNDpbOk_wl-_TnHhPzConJzrQ5Tjq-10tvMfPfomf0PbGhxcrP5MDyA9vUX4Zn8-5czMnAgFTmcr_1Br75CN_YNoM1vraD4oOP_p7OOvAD9oBbbyzwxTamq497RmF_8Qf-S01StM9vBPjVg0_8xMO_h_M5Baf00boOZuV2T78AMY0D6L-18XpUg_GwJZ0l-BfWpSUpKSD_NnUf9o9or81WyAmj1QkuKU8mudAPP4Sv27dSsstIK_cnrV_lo2wDx-RP9CfyntY9hfpb0Wf7UbYB4_OTk5RX3M-Sf2vpbR_mo2QGCfrDYN_uFHQEDfIVVL0OqvdAOs28fiH3IEpPSdizyGxMRY_JVsgHX7mP07joAdekXF6h9tA6zb6-MfNAPr0Ovm76sHnX9M9Pr6-7MKvH7-PZoL_EIx1z92ev39jViAAe56-Qfjt2WCv070RvjrtQMD1fXhD2nfYQZsyxvqr3YIZnjr5x8Nv12Mwpvk31aX1ijJdfBXZR_zFAxzp_NvjZpfg38s9irWYDg6_FX760sfyEzmsMGfgh7-1vA3ih7-jPsbCs-Qv2n87PmTucOfKX8zyeHPjr_54PCHP6_-nfjhz8IA6Pyp-eHvD_7whz_84Q9_-MMf_vCHP_zhD3_4wx_-8Ic__OEPf_jDH_7whz_84Q9_-MMf_vCHP_zhD3_4wx_-8Ic__OEP_4j88E9iYQBk_tT68Ic_edT68KeNWh_-tFHrw582an3400atD3_aqPXhTxu1Pvxpo9aHP23U-vCnjVof_rRR68OfNmp9-NNGrQ9_2qj14U8cNb9Z_gzywx_-5FHzw582an7400bND3_aqPnhTxs1P_xpo-aHP23U_PCnjZof_rRR88OfOD78WeSHP_zpgz9Z1PSB4E8WNX0g-JNFTR8I_mRR0weCP1nU9M3x4M8kP_zhz0Dwp4pavjn4U0Ut3xz8qaKWb4lDf2r55qjhW7K_P5vf_vCHPwvBnyhq-Nbs7s8oPzMDgD9R1PAtwZ8oavjW7O3PLD_84c9E8KeJ2r0tO_uzyw9_-DOSff0Z5mfIn24AHPPDn-9vf5b8qQbAMz9T_kQD4JmfLX-aAfDMz5g_yQI41mfPn2AAPPOz52_-AjjWZ9I_yeQJmKNPDR0maumwWZrfIviJDPv7s6S_dez9URNHzzr8lviF3yFqXRUx629B9mtRq2qOkt_K4B2iZjQ0Tfr2sVUWNRJZ1IdnKmoMM6O-NTvFdehn20XX5MsTvYvqXiru_Xi827tJlO-ed2_5GSm5oWr4R0KP3bd5nvjQKf3wsigPPJgpFBzNLe0uHfJUXxHlca9lCul7t9TdPKiPlOJ7_-7-ucLnFas8C3dkS6v_J8onfG-P-pso710eX_7s-YfGdH28Saj_QHSdmtPdO0Ec0VBzW6p72YfjN02YcUl6_5Qo_-jrXsKiLKn04LgLnsG-z1eX1ENYWz_LdU_WqbEPfZEg1dZvcz91d2bj0m9Xy-8U1paNHVRZ8dlZ0XWi-j2p8qfbG5YU_EN4cvN8aWp8nPTb__ierzBHWPD9-z3DHrur9NarojyjarAwaJ3oWjx0hvfKWwVjKg-dFM5JVfLKswMaL714X-mQnCJh8ZGrngPz13ju9P37B57LFKYsaJTyrmZI-86J8up1uULB6_WNXz85Tz67ZUnZulfKKvq8vLxk7vzc8qmHRfn80hTh5S8uln35V1F-a25t2cxuPSs2vCnKO6YNKx85YkPxRxc-ER6Ynuy99chKzzPHRXn7xUzh6OsvSE2HS6RtH4vy-oFZwoRPRdf-0_ukbdKbDZdXvi1smJld-sCVJd4di0W56MESof4HfcoSVtdKqZdEeeauIcLGplljL8xZ5Ip7-hHpvslnKiZdTi1ryj7uKfDd_y_ZxUK_FYLcv36Y64SYL-0ce0fF_Q-L8uj51eW5k1c0nL0r3Z3w61-WrBiR6n3hqCindp8i5C9YVbqx_7-8eyfVyN_LE4XqLQulyoNdvFPfFeVli6cIKQVfeR5-9Y7GxV_9Qv7Tt_YL004MkW7fvV76-XlR_iTdITQur5LvXTZ07KT6OE_tgk_djyZelP57YLX0xhHfPiY6hd8cSix7LOu0tOi0KL8xdKNr65H90oQDmz09ffe9cKmL8OOTvaVj92z1DPc9f9XpNOH4ow9K17ue9sz5UpTdrzqERXN7S3sO7vSuyxDl1_74vGtm5Xulmz8XXYn7aqW6V2685eQJ0fXSyARv2dCfNFQm5bkHl6VJC9ZuKpeHi_KVcY8Ik5-dJ63aXVi-pVGUK27fIdQ4T0nbt-4prztdI5-Z_o4gbWqUirwfeJt-VSOX7_q9IP7hd9Lh_sOkY76vb_a08cKMf35W0iv_-fI1A0R5uvM595In1kvS8XPeBleNvPC6YnfO9I3S2rrveHc95dvvv29y39L9RckxOsX7_gFRHv73P7vEibul8WuWekd8V5R7T4pz72mqkt4-Vlw-e48oV-90uZPfLS6rnbXUWzhbdNV-nCT8H5Kkoqk'
    s = 'eJzt2nlYVPUex3EQQcFwC2VRVHJX3FITmO3MkEvuplmkZqa5oJXbo5XiUSulMCvNLG-ZKURPZblrzhl-5c3nJrnca3nLrJte7Wqm1_Vq7vfMDDAL5_zmzPo9zO_7_icVIub7-syhp6eaORER1SIia0ZgGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIZhGIaFS9U8Bv0dYsHIszsOITzzQR5XEB75SY8jqLoFjB43UPUKPD5uoKoUNHucgPoLNj5OQM2FRh8noMpCiI8LUFuhxscJqCkYfVyAOoLTxwXAB6uPC4AN2t4e9BVYDdrdEfQl2Axa3TnoWzAZNLpz0LdgMWhz16CvwV7Q4m5Bn4O5oMErBX0QxoLmrhz0RdgKWlsq6JuwFLS1ZNBHYShoaumgr8JO0NJyQd-FlaCdZYM-DCNBM8sHfRk2glamBH0aJoJGpgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYFfRsWgjamBX0bFoI2pgV9GxaCNqYGfRwGgiamBn0cBoImpgZ9HAaCJqYGfRwGgiamBn0cBoImpgZ9HAaCJqYGfRwGgiamBn0cBoImpgZ9HAaCJqYGfRwGgiamBn0cBoImpgZ9HAaCJqYGfRwGgiamBn0cBoImpgZ9nLColpj8R6GJqYXuSGFarbLkPwOamJryF5rg_63CrVpOyX8WNDE1pS81IQH93avFjH8C-kvEjH8a-ksVjv4NGzas_ArQXzJF_hkZ0MTU3L_dhlIDUOSfnOz9Bat2Yeh_r4_-ydZ8u2JVqoGY43cM-nfvLv0qw9O_l_sfsOGfFdGtm8ufMuff2f4Xd38O_cPEv5EY5cOd27aNGG7379LFYe6Df-PGjaGJqbl9u-X-vQ0Gpz9NSxha_vM_jPwbNUpLS7u7PMfH-kYMEv3FBUT00ukeQH82_MXEp35mJvo7-_d7sF-E1X8g-kcw5T-4wl-r1dr9tdbC2l-vz8zU6_vq9XZ_vb7c32QvXP015T3k6v9IirUBA-z-A8Rf96goJSXbnvULJFcF__ZOSftHR2dmRjtqac3m39Kezb9lReHgn9TRJbt_x269ReukpJSy0hJSRP9OLkX0F0uSrF27xMSmzjVrVnGP-vXr-w0ZJVF8ebGxTZyKj3f82mg0dihP2r9Dh0zrB2PKsg4l3jmb_30VuftHRt5vrwr5p7hm9y__XTloWkKS6C8pbS0x0dVb5GjmVHRsbH23movF1KnI_na7y15UVA1bTkeMq11eTF1HUjNw3YJtDtasmi6rcPFvUsZvtOb0_q9eOdHf8RsX_1YSVQH_rk6Jh7L5205m07fqtrAl-rdIrMjFOsrdW8z2JdzQxT-2ucdUko-Ojomp0C_3txYZV574OY4RxCiaQeUxOAbhmIVbZftQlIu_y0eGDLF-rTZiavdv5yC2ZfUv-2VSUmJTp0MmVPzKRbosx1EroTvcnemjHfQxLvou_pGOBcTZPtt5BF7tQGYOUrOgzMMlF_82rpW9UWLV7k-7VWK0XG6HkiJ3ZZeUL7N303fzd55A2d_ltgGfdqBkE4mecvZPlFpQrOr9W1eu4nt3Z5YRr7QOZ3XJp72zvTu-hH-lh4DLCGo7_7P82oF78i_fLau_zIfU7i_7Fne8y2Wd3dFd1OXe8i72UvpS_hIPAdlHQeCWoNhfunusqd2_j5isP1280htdAbwzvYy-jL_kQ8DTCPxbgs_oYhVfROX-0iezV0mb9sme3N3pZfVl_WUfAopW4MMYfCV3C5qYGtXfo7e7u_yjotLfIodP9ac8BLxagdwmPPrf4wnbtVRr0MTU5MUkzLxWl_0qFH26v4eHgB8rqJwyY4p8aqr1m4EmpuatnDJzDwOi6nv09_wQCNAM_IUvC5qYGpVPIbMX8p7xlfh7MQF_duCne3nQxNSoiAGWV6avzF_ZzwH3vByCX-wVQRNT84VYOiUHVYCv3N-3CZQn_d-PFPkrMHcOmpia94dzT9lbKUbZW99Lfy9_DijJsYsAfcEw9VfM7tVb32v_YEwgwEETU_PidXgr7stb3xd_tU8AmphaT1sy37kf4j7j--Sv2gnUE4MmpkaFCJy-cnxf_V02AM1ur549aGJqwfX3Ad8ffzU9Bsrw61WvDk1MLYj-vuH76a-Kx0A9B351Rv19xg-AP-xjwMWeTX9_8APjD7SBSvbs-Xv9b_u0AfjlH-INSNoz5h8A-8D6h2YEsvQs-QfIPgj-wRwBnZ4V_wDaB8s_4CtQIM-Ef4Dtg-vvvgKfZqBYXv3-dAcA-pD4-zYEL9mrgr8HBwD60PrLrsFWpLfSVczfM4SsfJDgYf0rFd7-6d75h8C9vAD9ByD0p2R7gem25CFC5O0e-gc7yZebnu62h0gYfvQPdkoPgP7h6K_8AOgffgPw5gDoH27-3h0A_cPL39sDoH84-Xt_APQPH39fDoD-4TIB3w6A_uGxAF8PgP5hsAA_DoD-VX4Cfh0A_av2Avw9APpX5Qn4fwD0r7ITCMgB0L9KTiBgB0D_oKZu_Ej0D0GqtbeG_iFJlfbWgPzV8j-Ahsrf3xEE7QDoH-rUIm8P_UNfTXuQ7BWhP5i_GOyLt4b-gPzoDz0AWH_Ql24P_dEf_dEf_dEf_dEf_dEf_dEf_dEf_dEf_dEf_dEf_dEf_dE_8AMAffnoj_7oj_7oD-YPOgDW-dEf_dEf_dEf_dEf_dEf_dEf_dEf_dEf_UMxAMiXj_7oj_6w_pADYJ0f_dEf_dEf_dEf_dEf_dE_5AMAfPnorwJ_wAGwzo_-6I_-6I_-6A83ALiXj_5q8IcbAOv86I_-6I_-wAMAe_norwp_sAGwzo_-6I_-6M_qAFjnR3_0V8UAgF4--qvEH2gArPND-0M_ANBfLf4wA2CdH9wfeACs88P7w_4EQH_wQB8ArPOrwB90AKzzq8Ef8kcA6_zq8Id7BDCurxZ_sAGwzq8Wf6gFMK6vIv8okAmwjR-pLv-o0E-AbfxI1flbC-EG2La3Ba0tV0hGwDa9LWhnTwX1acAue0XQvl4W2DkESDwQ3wpY0KIBr4bijYS9raKgvcCCPrzKguYIWdCHVlERbh3dVVS4_qFczYo7PNndOYW7MHSeIa_gd8v0ji-ZLc3u43pMyhV-HviJZsKfPIm4nMTlL9xoFvZM1Ky5zZNHxM8v3nnF_O6SbprfbvGkN9-EW_Vld2FM6XDN6hs86bo7lbsrT1t49tQdS87tPNL2qSTuQJOt9xQVt7DU-JYnK44mc7P3rhI6TVutyb_Ek2GDUrhRC3ebO67P1bVZzpMrMztw-z-dY8jK0ugXau8SCu7kG4eMXyAMXl6o-6k5TyZXu24o_G6J0FfXaseQKzzZvrIhN3BZe6FRMaf7dTNPvpiQwUU3nWHOPTbKMvR9nmw8NZg7N7VAIKkLdQ2n86R40sNcre8LzTcmXxZuH-PJD7fjuWHPp5tH58SWLPs8jxwcvZX7fF0n0lqbZTiXedrc5cvWponHxe9r4hlhX9f9mnHfHeRyP-UN78YM0KVt_tFc-8poY9v2zxtmDMgpKRqfZt5zc6LxyNGtwuGpazR1zvHk8MPJ3G79YmHJgUTBeJkn-pv9uJ7deRL1-Rrd0Z0rNd9fqme6dfFhYdPVjsIy8fWkpT_KPXWVJ0tfeVab1XSHJnfPS8YxAm-IPJhhGXn8I3PTjbOMxTHVhY0v6y1nt_Fk_-T_GsZuXiJMm6_VHd_Ak7Nz87kDxRohLX6p5cgongzuXMwdOrXB_MGTsy3jFou_1y_jdg6eRzbHn7YsuX5V09LY3Vh0ZIpgaTVceFX0iK0Wy13711DhjW0zdOMXia97_BbDi5veNN9raa09IfrG7anFjbkwRth1_BV928dnkTYzok0R5q3CnPwY_afb88i612KMuv4688qTNUse25xHaqz4lftiT6Hw1zOv63IH8uSPJcu48y03WC5H84b93acLPZ6Nz24x4lhm_wZ_s6zeM49czDvNvXdxntC6xf-0O37iyczWy7hGzxUI26em6Ut3zSXTj3xt2DsiTnh131hd1tviHXNKuE29C4WFnXKEumd5snLRM9y1s2d2npz3sWV8Y568Oa2-cae40219iLnJll2a-AUtjQXGZMOi6gMMyY8vEGYnEVPfHk-SwtPHSy7v7qkd9N0zxuJVv5mfbiBYnv59Hnn79drG5q9ONux94ZuSF0nzrJqp2aaVJ3ny53uHhQ_q7NNkdTlhnD3lM0Hfo4VumsCT0Ru2czErqmk2D4kr0RbnkR_qvWDM-I03DJt5Qttqxz7z_YWpprHnxTu8-Inws7FU89jYFsa2BeMMez_7T8nAJ37Jyj-ebOr9wCXt0zlvl3R9YCaZNGa46d-94oTp17qUdE2aSy5Expma1Ysv6Z-fZyj6ZrLQckj77IyXDcK-h2L1Ez7JI_pRpcbb454y5L1PSvjPijIbPTHDNGUQb1j12grd2uvHzV8fW25cnfe-8NaU0qz5onfetyncgQ8XaDYeKtLWuciT88aV3KERDbOy539l-ftP80iTgu7GRR8ZzUdSP9TqxI8_t3Qkt-g6MV_SzbaMFvd07YqJO1CQbDi3t79h24MLhF_GJmd3GPMYeWPprZK1-wZp7z_c0bT-4Dvm_CnZlqIt4r7IJG5xnVzDmn4H9IZZOuGj88tMs0W3Ng0E4UL2t5rt_5hv_DidJ1s2r7NMzjmp6Tn-Q2ON_X-aI0Y-b14rOqb20HD_fGuTOfOddywzevFkZEa68UbxMLJ_Tm1ycMUIbavoSSa-_wBD5okkQ534scKtS9-Y1o14lBSsiCYJN4ZpnyQFpld6txem5vax_LGeJ_0W1zU0vjqHfHW0W4l-T5z29VF1TI-kzSXV2nUpWXsuVkv6pJkyptUVoqJOaseJz4E2i-82_HhmtnZSTrTlq0Pi95l4mpt9ZIu20ynekDEtTyi90zb70qot5q5bZ2mXi-_jmx3mc7qf_yKU1phgKS0Qnz836xr_D_uhnYI'
    s = 'eJzt1n1wFPUdx_EQw0PCg8QmGJBQAQvVgjwZYHKb3F4UbCvDOEAFwQpadexYRI04U55WMGBEsBadQWttKcxUQGqZ8CTmLmulrcX-0YrWGesTCtrCVChiUUCkucsFcnd7d7t3v93Pb3c_rz8yJJBw93l_c0m3sQUFhQWduhUQERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERGJdUF26IdIdjARnnfgSdbL8wo8Ir_0PAI3E9WeN-BCouPzBtzDpvY8ATewNz5PQGpOxOcJyMq5-jwB6TgcnycgFUh9XoAcYPF5AhIA149CT-Bj6PRx6Bl8Cp29A_QUPoROngQ9h8-gcxtAT-Ij6NRpoGfxCXTmDNDT-AG6cWbodbwO3Tc79EJehm5rDnolz0KHNQu9kzehq1qB3sp70EUtQs_lNeie1qEX8xJ0y9ygV_MMdMhcoXfzBnTFfKC38wB0wvyg13M9dMC8oQd0NXQ8EbALdonDPoocodOJAZmuS5Ku50EeTw7Q3cRxcrXk8K7tj44mkiODpSnvzv7oYqLZPljXruwvM7sHY3-5ldo8GPvLrdTmA2B_uZWaPoAePXIZjP3l1tq_V_an3SMml8HYX27R_h0PYFDqc-7RLpfB2F9usf6xC7goKmP_nnFWBmN_ucX794r3v4j9M0LnEq69f0UF-5uAziXc-f4V7J8dOpdwfujfv7-wL4XOJZw_-vcXcgLfGjgQnUs49jdtCPuzv9dk7j88-sbv_Su-0cZX_UeMaH0zPKF_Tz_1v7z42-1_7ND_MnQu4djfWGv_9gPwX_-hccOHjxo1qsrt_YvbWPukyy_3af9o98IEKf2vaGdiR3n6J8j6Sb7tP3r06MIs_S8xZrijpP2LL44y-tcjR44suMqX_WMRB0XfZOlfmFbfqLKOpO9faeCqyspo__h7PunfMWPm_n0TlKVXUlIidf_BifoZ8Ev_hKSFCe-190-fuqSjb7bp1kbq_v36DcgsdgI-6J_0Mt4tIWhJvH-JscTeycz379RBUQfdz-vcJy768XN9Y_9q2LDU8OVt0vcvz6z9NcDr_VOrGWSO9s9YOkXslHLrn3QCCTfQqsMVxBQWFiUaFnVhqytbmel_oZGkc_Bqf7M9zUfvIOf-SReQcAOxI0i5gtghFKbcQlH6_obV4y414MX-4pOL6p9yAQY3YHwG7ZfQdg7p-6d8_fOMj8J7_cXWFtzf4AJSfxgknkHyHWT8_jfR_ztxbc_Ga_3TNC6xGNm-_sYXYPxCkO4O0va3_mw81T_N7_NREvVPdwEZjyDh54Jx_wzf-xmgm4mTob7I_oWFeffPcAFZjyAqNX5u7b3UP2N9sf0znoC5_pkvINsRJKTP9IuCb_pnqS-8f_ozMNs_-wmkv4Jz5S3w8AFkjW9bf4M7OP_BbPlNXUCGK7DGq68B0aXR-Y2Z6G_2ApKvIIczyPzF0RVzhehqlqn-lk4g5Q6E9XflCaADZ2G2v-ULMDiE7JeQ9ct07ozuaQ06bzbm8-d8AWluwfAYjJMnQ0c1Cx3XBEv9BZxAFqZ_UKDTmoBOa4rV_jZfgOn-sp8AuqtZ1vvbegJW-st7AuioFuTU374TsNhfyhNAJ7Uk1_42nYD1_pKdALqnVXn0t-MEcuovzQmgY-Ygv_7CTyDX_r1792b8HOSdX_AJ5Bo_JvZg2N4KIf0FnkCe8eMY3yxR_UXdQP7tnT0BdL98iewv4gSEtHfoCNDtRBDcP-8bENfe3htAdxPFhv55HYHY9uewfRo25W8jun9O6Ttg-RQicxsT0j_f8glY_hwhe5qSa3-R4ZP5OHycLbNmZqq_jdHT8E_zjpxcWG7oEhjo1eWBLoGBXl0e6BIY6NXlgS6BgV5dHugSEOjRZYJugYDeXCboFgjozWWCboGA3lwm6BYI6M1lgm6BgN5cJugWAOjJ5YKu4Tz04nJB13AeenG5oGs4D724XNA1nIdeXC7oGs5DLy4XdA3HoQeXDbqH09B7ywbdw2novWWD7uE09N6yQfdwGnpv2aB7OAw9t3zQRZyFXls-6CLOQq8tH3QRZ6HXlg-6iKPQY0sIncRR6LFlhG7iJPTWMkI3cRJ6axmhmzgJvbWM0E0chJ5aTugqzkEvLSd0Feegl5YTuopj0ENLCp3FMeihZYXu4hT0zrJCd3EKemdZobs4BD2zvNBlnIFeWV7oMs5ArywvdBlHoEeWGbqNE9AbywzdxgHoiaWGjuMA9MRyQ9exH3phuaHr2A49sOzQfeyG3ld26D42Q88rP3Qhe6HXlR-6kK3Q47oBupGd0Nu6AbqRjdDTugO6kn3Qy7oDupJt0MO6BbqTTdCzuge6lD3Qq7oHupQt0KO6CbqVDdCTugu6lnDoQd0G3Us09J5ug-4lGHpO90EXEwo9phuhmwmEntKd0NVEQe_oXuhyQqBHdDN0OwHQE7obul6-0Pu5H7pgXtDjeQG6Ye7Qy3kFumNu0Kt5CbqldejFvAbd0xL0WN6ErmoSeiYPQ6fNDr2Q96ELZ4Cexi_QnQ2gJ_EfdPFz0EP4GcNTJ6fPAP1sKR02p3bsTIaYnoiIiIiIiIjInwqS_Hvf4CHbRk4MbDur6Z8cqFZn_HNWoOHio-GdBzS9941BteaMFmxctjEwv-pkYMhNe9RTC05Wrzt1InzxR5q-eF6V-sb7k5tvnl8Ymfu-pu8dO1qd-aamH_7FRTWrph1q_u-w6aH64Zp-6YANkablJeF377onNGvwj5qVs_vDH32i6c33XKPedmyRXrqlsvbjy3Y339gwPnR60KOBX_b5MLy59e_7zrhGvaumpnmZVqf0P63p2__8A7VMXaBXrZne8sNrB4bfm3N16Lq1c2u01Zo-6Gfjw4-eLa07eFIL1pwZr7z12OnAX9f9TX2qsky5fsyD4Un_0_RZq2aoL_7x0-qNT6yIhO7T9M-7DFWHF2nB-mVNNZMOdFcab1muvly5puZf0zRdaxkbXrK1sa7ot51rr3htcXDHysnKW8_fXvdY35bAiYXHI-VXLtEfn7tHfWFXg_Lhqzcp9cc0vajpGXXt3Ur11peWRm5epOlTbu-udml93KNKBio96483n_h7r9CT72jBlTM7R55_-nBg1d0lod9N6RLetOPr8IDW_e6qnaQu_ez-3Z_MXLX7bGuPG378kPrdG8uVB6ZMjpRt1vSJux5Rt09dpL-nD6vderCp-d4dB0JT5jUEbv6gVilq_X9qntug9t-zOnBJp9_UrL5K00du6heqe_3h5p_OO6S8cVDTZ5buDb5eu1xZOW9FeOkRTd8Zmq5OUVcrG49pwS1D65TP_tSn7lebFutv68UtC6eVheu3L1df3XIq8EFxU6DyjKY_8-QadZyyJNg0-2jNqXGlyv5tV4R6TVgcXFtc3nKoKRxY-84udc6XWnD9-9OVFyd-FfjDaxNCG145qNzx9L2R6-7X9DFVR9TfT2uI_Oc-LViyeZzy0itv192qjdW7PDwmeHWoOlAx9aG66fvXK3fM3h05_fES_etBX6rr3n02cOecqprPWzR939Hn1H2DtwZeXtEQaZqv6Y2HI8Gq9Zr-hDYjsvepzuEXtjeG6kcrygPjFka-_6CmT5iwOaiVLtSL_nJty9SdA8KT3imr2xXpGT787OPKDcc1feCRf6h7d25VVm36Irx3X-vdLvieetvPb1Uqrj8a3vOeple-GVBfqtjVPLT4AuXAV5pePvs59dMeC_Q9P5lVu2DEr5sHdn8l9Papk8qdjXMjg-s1_ZbqL9T_A0fneoQ'
    s = 'eJzt23l4FOUBx_EkS0iAmIPDhDPkMCSQAgIGk53ZmdkoYDgjl6AIeGCgtYpFSjTJhIoIT6tVQASNB1ofgSI8rUBb9nit0pZapWnVglbl8EBAwSqFYlrp7GaT7H3OzDub-X3_QBLDZvb9_HayPD6m3pqQkJSQmJqAEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQClxX97p0RPu6kJJ19V8Xf9G-ViRfKVIB7AP6YwbxX4pbUfpjB3FZik8x-mMG8ZQvv0z-GEBcBH99B399B399B399B399B399B_9IS01NpX0JMqYT_-5SMj1UqkcyPSi1dODf3ZVMD5fqN5keXPUU909Lo_jsursn02P694-XHTiAvT9W2J_OArr7JNMDB_fX8gxcvv4-p5j_8DQKA_C1V9s_NbWfTN9Ollr16PmruIBM__ja8e_lSKZrCSM3PZr-qiwg05FW_AMNoFdbMl1PkLz06PqrsABt-QcYQLt_To5MVxSgXG892v6KLyDe_LOzs2W6KM8KCwsTNOmv8AI05u9_AJ7-2VfEej1ZWV6fKCx0-udq0v9yqVifccDi0j-2AaSkZHnUqq9tfxk3ILHm5eX1d32kNX-_A_Dxj2EADi5P_6w48ZdlAE5Wh78rzfn7G4Cvf06U7wRbuQL453Z-_-7a93cIe73H8-vvLMFYEMF1tHHBX-v-ma25_nxg_xxjQUHYCxgT3D-3s_sXFxfHmb9rBCH8w10A_OEf2D9B8_6DnYV76Pk3D-nb1-MzOvcfE9w_N078zWE92fz8IUMkf4_gH9Bf-mS8-48e7fzHkCEjRybozb_EWYir8O_fbuiP35__tdrzZ0ZL5efnj8x3sMPff_L4X6s9fwb-evKf6PW8FsBfLf_Kykrt-TPw15e_1wDgr4b_pEku_4Xa9589e7aO_b_n2fQ5MvlXVVVVVkq_0PafNWvW1KnjpW68UeiIdfNnWVaP_i5w3pHHAqbPuekWkykG_7KyskmTylzR9pd0W_3ZCayjaROcTZs2rc1_5syZnch_6NCQ_jNmzJgi5ZTnudaqq6snS5nai8G_vLz8mvaU8x_mLJT_lV4ltdb2oeQv_RrKPy9-_CsqgvgP9cgJX-RoaMCi8Z-X5JZy_tc5C-g_zKe2W5JByvPfyON_taOePQe5ZxjYVvJlzvySZ7hK6i0Vm__Ysf78Kzoq8qybo6Jguf-ZsPzb6NMdKed_faLU_ID-5e4lJxs8SvbM4KfI_b3p3eyTXfquEj3KcEs6ttj8k5Lc_cd61807p5TPZwMVnn-6W8r5O5Cuvz6gf2Lw_JEbDOPacn5NhP5u8m72Dvxg-j4LiNk_SelC-Lvp9-6tnH8fZwH9_QN35H8Wc-f28crh3_b7kP6ux_aw9-L3_33dBhCrv6oF9u_dmnL-o5wNL_T09-aTL-mHe3B___bh6HssIG7821_l3v69O-qmnH9PZx7-o_wUu_rVPT0L7O9rH66-2wKi9r9KVXSf2v3b6D3fMSjy899gGN7-A6BnpEVi7sr9_Z2Pv699JPrtCwhL3_HYLn-P73eVK5XM3Wu_3bfRK-xfWto6AInez4_3G24IewCBsAPB-3yvDn-vv1eEeNvnfwHB4d3urcmhimIM4VH7pXd_1avhn1zaXqj3egEK4x4RjD3Qz_vo9Z35QfdC6t07khUE34T0qejMQ8or7t9RlEsIhz3oAwS2T4701u_u7_7XwYyAL9UYRuCZQvKOBrTmfW4d4F7PKUp_90ojGINf8_C2E5Q-Bv3WwltArPeC6PzDgg-C31qP1oI9tXZpr8MZMWJEdE_UOQ73D8OjjpQ-OVZ9nwkEXUDMK5AdPrR-WP5JSUGOJ4YRdKQEfbIs-t4LCD2B6HcgK7uHfpCTDIc_-AJaRxDTCqKhD_mgMulHuYDIhyCbebgv_Q7-0P6hFtA2AyX9w5aXWT_GCYS3hQj-449cL_3I_MNagGsFkc5ATnhF9GVaQNAUwA-lH5l_2AuIfAfB4SNckyL6yk9AffyI_SNcQPhT8KMeKbry-o4UXIC8-uHgR-Ef7QL85phF6--ixlYVvzWlJiAjfrj60fjLOwFX8aPvTJEJqI8frb_8C4grfWfyT0B9fEPYf_1XegHy6st7bQGTeQKx40doH_3LX_YJxNVL3y05J6DuC18GfxkXEJ_6zjLk2oDq-LH7yzaBeMV3JcsE1LaXx1-eCcihL8NlxFDstwG17WXzl2EBcY7vKrYNqG0vo3_ME-gE-G1FvQGV6WX2j20CnQXfVVQ3AnXl5eePaQOdCL-9yEagrrxS_tFOoNPhtxfuvSCAuxLsyvpHtYHOae9WRogdZBgU11bRP-INdGp7nzJ8C_A_Qsexf0Qb0I99oDqlf9gj0DW9s87rH84KgtKrA0C5Ts4fYgZ-4VU9f9rpxD_AFjrYqQFQTof-btE-ffrBX9_BX9_BX9_pmh_-8Nd58Nd38Nd38Nd38Nd3uuaHP_x1Hvz1Hfz1Hfz1HSV_2vCuaJ8-_ejww18rwV_fwV_fwV_f6Zof_vDXefDXd_DXd1T8abO3R_v06UeDH_7aCf76Dv76Ttf88Ie_zoO_vqPgTxvdLdqnTz_1-eGvpeCv7-Cv73TND3_46zzV_WmTe0T79OmnNj_8tRX89Z2u-eEPf52na374w1_nqetP29s72qdPP1X54a-5dM0Pf_jrPF3zwx_-Ok_X_PBXw1-7_PCHv87TNT_8lffXMj_8FffXND_8lfbXNj_8FfbXOD_8lfXXOj_8FfXXPD_8lfTXPj_8lfOPA334K-cfF_zwV8g_PvThr4x_vOjDXwn_-NEHv_z-caQP_0S5_eNKH_6JsvrHGX4S_BPl8487e0e0D18DyWcfb_hJ8E-M3T9e6Z3RPnwNpMNXfUe0D18D6e417x7tw9dAemRvj_bhayDdmbtH-_A1UIcybQwK0T58DUSbgGq0D18D0SagGe2z10K0DWhG--w1EW0EitE-ek1EG4FetE9eI9FmoBXtc9dMtCHoRPvUtRVtDXWjfdrxEG0j-aJ9klouwauHU-r2PX18nLWwRSQlqxi-YH2xMWtXqnXl_0RyrziGX7H9qNE4mLXmfCuS5pFGvuJkLXnbPs90ZmepNbtlmTAk_Urm5wOusB7-j0g-2MXyE3Y3kLt2v8l2-1GJtWluo_CnN0TSsjOXrTxcZN20Y6Vg_0rksrObrP0-uIKxzBKFv792OzOmz3eWjRdEcs-yUfwuUs89VJRi2mYewBgL1wr7v19TmDb8p2x9jUhuTSvnv9yxnPR_f439dzdnWk_WFAifjVhg-dt3T7DHqkTy9bo5_OmUJuPSlIPWh06J5IaGCfzaXnuNn_e7z_bKKpHM3juZ73ZE5JoHnLdetbiIufCbBcKd7_W0N8ytJ2czv6poGFNsPlS4kTmz6bmKb86JhJ84hf_FJ6O4T_uNJTta-jLjuYvC7-_LZDb88knbBLNIruqymJ-9rM74i3-VMT-Uzq9p3UK-_yCRfLFhK3vHa8XWqQeeEfbuELkTd1ax317MYwbnzBbKRq7g5r95n337gO8x__ztr4XNq--217y7gnvgH3sty9hJZsPwQez2gyJpKLrHmDt6rrDypoO2w79q4MbnPGx58K-fC_nVyczx6v7sr6WvIWwxv_Wb8UyP7F6295pF8vzKYfzRm20V015cydbXieSzlx7k6xYV2t94uY7w1YcrbszaJzz2zl_tV5xbzCWd-dzy6R9WmY9dM5A0t1zHpadnWndevFtYu_JujszbZdpSncr02_a-8PyhOu7jb3PtOZOHMXXVLwuL9tSS789baGqyl1ofG_yQMNFYwdwiPXfL9AXGV5-83DytdDfzaPVuS_FZkdRtKeULJtVyls01pnSmN7P-k0eFGRl13NC9o0yHP8xhXvliq_DIj-y2dz5uIBs39DMabj1q7rJkFdOH_aXxp-dFclldBj99eZNt0bYxplvK7ydnz9QLtcxqY-Xic8yE4yKZnFvDdyHPV-SLr1r_-aVIyp7axtcMz-PePT6Ou2RdblyWPpB_ZuzjTEmfHPuVg-rJc4v3cOzl87ljA1tMfUpOGQ_0KzQX7TdYX582w7hW2jt_bSP_9nPP7mvc_xQ7SPIde3I7P_fKBntTy4-5Y0f3WXLu_c68KsPAGj4Sif3dO42bblpknrK7K1MwM4td_Y5IEu3JfPWDObaiy69mJzwqkp9vPcp3HXDAuPtIs63btgby2ReEHzDhd8yqhlW2k7dL13_bEe5ndSuNb_XdYdz4nfT1zSJ_Ke1-5qXHn7ctLREJN3YP_8LUOs7wdampZmZfZuKBefyCYwu5hae_Ml06ddb4k0-GmJvuX8PsOXCRueuwSCrzCN_j8S7s0fUTrZulj5eKd_Mf1Kcxm17nmfSLIpn-1Hnu7YwmJqF8s-W2b0SyZ9ZUfvW6I8x1b-1gEt4TyYdzHuZ_sHAp2V-_3X7u9m7WQaPX8RuTfkxuW7LafuZQprV08cjK_9473jY4bT2z8yWRbCnrKtyZd5bN4hpI1fB1xmuWVJs_zLjHsrZMYC5J94sTh5fzuVPmSeeQwE3eX2I9l_aI-Q_XlTM53Q8x5Z-KZM4be_jyPR9br3nayvZobiA_-aKPULUm2b5iwx_Z05vqSPaxNcKXn0vnuuTP1iW1BdZ5eY1m4dIF--nz80nl3kRL4lJD5avDCthFBx9hDK-JJP2B3VzfCznsmZcnGS_8TXr91wziEx9-jM2cK3KGH5ZYdlbuMH88f5P1lvMb2RscvsXDhRbDLqam11Tm4GmRpOWc5GZs32BNmEjYl_7RQMiBo_zMxh7sZac_tJ76vUie3PMsv7SRMNONTbaeI0Qi_jdZmLtlm_H0zL8wZ6T7zUcT9_JvLf3IMnBFlqn59nry5iS7kF-bva9262esYX4DEQ__TOBWnWBmTPyPbUxSA3mg-79Nu17JNd2Y2mI9Za8jQ04M5POzn2SY-ZttsxmRZF-o4t5a3oc98OJ6ZvefRfLMHXfwLxRbmEOHdlrPH5Gud-dT_AtT0mx3vW6oWCP55R0_wc944RX7tN_exY0b-57l_Wdmm_mqmcz2g_NtazdJ98eMRq65tpF5-oLIfV07xbLlN_0rVzaOsxe8mGk6tXoFWVtChG7CE-y1rzVZ_8eL5N_NU_j_A-JC3kM'
    depthMapData = parser.parse(s)
    # parse first bytes to describe data
    header = parser.parseHeader(depthMapData)
    # parse bytes into planes of float values
    data = parser.parsePlanes(header, depthMapData)
    # compute position and values of pixels
    depthMap = parser.computeDepthMap(header, data["indices"], data["planes"])
    # process float 1D array into int 2D array with 255 values
    im = depthMap["depthMap"]

    # print(im)
    im[np.where(im == max(im))[0]] = 0
    # if min(im) < 0:
    #     im[np.where(im < 0)[0]] = 0
    im = im.reshape((depthMap["height"], depthMap["width"]))  # .astype(int)
    # display image
    img = PIL.Image.fromarray(im)
    img.save(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\PB03JZMg9nKvRjwf_T0xBg.tif')
    plt.imshow(im)
    plt.show()
    # print(im)
    # plt.imsave(r'D:\Code\StreetView\Essex\t\img.tiff', im)
    # ------------------------------------

# print(features.check('libtiff'))
