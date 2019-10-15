"""
Designed by Huan Ning, gladcolor@gmail.com, 2019.09.04

"""
from pyproj import Proj, transform
import multiprocessing as mp
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
import numpy as np
import struct
import matplotlib.pyplot as plt
import PIL
from PIL import features
WINDOWS_SIZE = '100, 100'
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
Loading_time = 5
web_driver_path = r'K:\Research\StreetView\Google_street_view\chromedriver.exe'
#driver = webdriver.Chrome(executable_path=web_driver_path, chrome_options=chrome_options)
#Process_cnt = 10

"""
Read Me 
lon/lat = longitude/latitude
The current objective of GPano class is to download the panorama image from Google Street View according to lon/lat.
Main workflow: get panorama image ID based on lon/lat -> according to the panorama id, get tiles of a Panorama image and then mosaic them 
Please implement all the methods. I have written some tips (not code) in the method body to assist you. -- Huan
"""

class GPanorama():
    # Obtain a panomaro image from Google Street View Map
    def getPanoJPGsfrmLonlats(self, list_lonlat, saved_path, prefix="", suffix="", zoom=4):
        """ Obtain panomara images from a list of lat/lon: [(lon, lat), ...]

        """
        statuses = []      # succeeded: 1; failed: 0
        #print(list_lonlat.pop(0))
        start_time = time.time()
        Cnt = 0
        Cnt_interval = 100

        while len(list_lonlat) > 0:
            lon, lat, row = list_lonlat.pop(0)
            print(lon, lat, row)
            prefix = row
            self.getPanoJPGfrmLonlat(lon, lat, saved_path, prefix, suffix)
            Cnt += 1
            if Cnt % Cnt_interval == (Cnt_interval - 1):
                print("Process speed: {} points / hour.".format(int(Cnt/(time.time() - start_time + 0.001) * 3600)))
        return statuses

    def getPanoJPGfrmLonlat0(self, lon: float, lat: float, saved_path: str, prefix="", suffix="", zoom: int = 4) -> bool:
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
        #PanoID, lon_pano, lat_pano = self.getPanoIDfrmLonlat(lat, lon, zoom=4)

        adcode = self.getPanoIDfrmLonlat(lon, lat)

        if str(adcode[0]) == str(0):
            print(adcode[0], "is not a PanoID.")
            return status

        print(adcode)  # Works well.
        for x in range(30):  # test for the size of map, column, x
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[0] + '&output=tile&x=' + str(
                    x) + '&y=' + str(0) + '&zoom=4&nbt&fover=2'
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
            except OSError:
                m = x
                break

        for x in range(30):   # test for the size of map, row, y
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[0] + '&output=tile&x=' + str(
                    0) + '&y=' + str(x) + '&zoom=4&nbt&fover=2'
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
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
                        num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[0] + '&output=tile&x=' + str(
                        x) + '&y=' + str(y) + '&zoom=4&nbt&fover=2'
                    response = requests.get(url)
                    image = Image.open(BytesIO(response.content))
                    target.paste(image, (UNIT_SIZE * x, UNIT_SIZE * y, UNIT_SIZE * (x + 1), UNIT_SIZE * (y + 1)))
            if prefix != "":
                prefix += '_'
            if suffix != "":
                suffix = '_' + suffix
            mapname = os.path.join(saved_path, (prefix + adcode[1] + '_' + adcode[2] + '_' + adcode[0] + suffix + '.jpg'))
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
        #PanoID, lon_pano, lat_pano = self.getPanoIDfrmLonlat(lat, lon, zoom=4)

        adcode = self.getPanoIDfrmLonlat(lon, lat)

        if str(adcode[0]) == str(0):
            print(adcode[0], "is not a PanoID.")
            return status

        print(adcode)  # Works well.
        for x in range(30):  # test for the size of map, column, x
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[0] + '&output=tile&x=' + str(
                    x) + '&y=' + str(0) + '&zoom=4&nbt&fover=2'
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
            except OSError:
                m = x
                break

        for x in range(30):   # test for the size of map, row, y
            try:
                num = random.randint(0, 3)
                url = 'https://geo' + str(
                    num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[0] + '&output=tile&x=' + str(
                    0) + '&y=' + str(x) + '&zoom=4&nbt&fover=2'
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
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
                        num) + '.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=zh-CN&gl=us&panoid=' + adcode[0] + '&output=tile&x=' + str(
                        x) + '&y=' + str(y) + '&zoom=4&nbt&fover=2'
                    response = requests.get(url)
                    image = Image.open(BytesIO(response.content))
                    target.paste(image, (UNIT_SIZE * x, UNIT_SIZE * y, UNIT_SIZE * (x + 1), UNIT_SIZE * (y + 1)))
            if prefix != "":
                prefix += '_'
            if suffix != "":
                suffix = '_' + suffix
            mapname = os.path.join(saved_path, (prefix + adcode[1] + '_' + adcode[2] + '_' + adcode[0] + suffix + '.jpg'))
            target.save(mapname)
            status = 1
        except Exception as e:
            print("Error in getPanoJPGfrmLonlat():", e)
            status = 0

        return status


    def getPanosfrmLonlats_mp(self, list_lonlat_mp, saved_path, prefix="", suffix="", zoom=4, Process_cnt=4):
        """ Multi_processing version of getPanosfrmLonlats()
            Obtain panomara images from a list of lat/lon: [(lon, lat), ...]

        """
        statuses = []      # succeeded: 1; failed: 0
        pool = mp.Pool(processes=Process_cnt)
        for i in range(Process_cnt):
            pool.apply_async(self.getPanoJPGsfrmLonlats, args=(list_lonlat_mp, saved_path))
        pool.close()
        pool.join()


    # Obtain a panomara_ID according to lon/lat.
    # Finished!
    def getPanoIDfrmLonlat0(self, lon:float, lat:float,) -> (str, float, float):  # degraded. No longer use it.
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

        #print(url)
        try:
            driver.get(url)
            time.sleep(4)
            new_url = driver.current_url
            end_new_url = new_url[new_url.find("data="):]
            #print(end_new_url)
            if len(end_new_url) < len(f'data={url_part1}{url_part2}'): # fail to get the PanoID
                time.sleep(2)
            if len(end_new_url) < len(f'data={url_part1}{url_part2}'): # fail to get the PanoID
                PanoID = 0
                lon_pano = 0
                lat_pano = 0
                print("No new url for: ", url)

            else:
                lat_pano, lon_pano = new_url.split(',')[:2]
                lat_pano = lat_pano.split("@")[1]
                pos1 = new_url.find(url_part1)
                #pos2 = new_url.find(url_part2)
                pos2 = new_url.find(url_part2[:5])
                PanoID = new_url[(pos1 + len(url_part1)):pos2]
                print(url)
                print(new_url)

            return PanoID, lon_pano, lat_pano   # if cannot find the panomara, return (0, 0, 0)
        except Exception as e:
            print("Error in getPanoIDfrmLonlat()", e)

    def getPanoIDfrmLonlat(self, lon, lat):
        url = f'http://maps.google.com/cbk?output=json&ll={lat},{lon}'
        #print(url)
        r = requests.get(url)
        data = r.json()
        if 'Location' in data:
            return (data['Location']['panoId'], data['Location']['original_lng'], data['Location']['original_lat'])
        else:
            return 0, 0, 0

    def getImagefrmAngle(self, lon: float, lat: float, saved_path='Panos', prefix='', suffix='', width=1024, height=768, pitch=0, yaw=0):
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
              f"&h={height}&pitch={pitch}&ll={lat}%2C{lon}&yaw={yaw}"

        suffix = str(suffix)
        prefix = str(prefix)
        if prefix != "":
            #print('prefix:', prefix)
            prefix = prefix + '_'
        if suffix != "":
            suffix = '_' + suffix

        try:
            response = requests.get(url1)
            image = Image.open(BytesIO(response.content))

            jpg_name = os.path.join(saved_path, (prefix + str(lon) + '_' + str(lat) + '_' + str(pitch) + '_' +
                                                 str(int(yaw)) + suffix + '.jpg'))
            if image.getbbox():
                image.save(jpg_name)
            #print(url2)
        except Exception as e:
            print("Error in getImagefrmAngle() getting url1", e)
            print(url1)

    def getImageCirclefrmLonlat(self, lon: float, lat: float, saved_path='Panos', prefix='', suffix='', width=1024, height=768, pitch=0, road_compassA=0, interval=90):
        # w maximum: 1024
        # h maximum: 768
        # FOV should be 90, cannot be changed
        # interval: degree, not rad
        interval = abs(interval)
        interval = max(interval, 1)

        img_cnt = math.ceil(360/interval)
        for i in range(img_cnt):
            yaw = road_compassA + i * interval
            self.getImagefrmAngle(lon, lat, saved_path, prefix, suffix, width, height, pitch, yaw)

    def getImage4DirectionfrmLonlat(self, lon: float, lat: float, saved_path='Panos', prefix='', suffix='', width=1024, height=768, pitch=0, road_compassA=0):
        # w maximum: 1024
        # h maximum: 768
        # FOV should be 90, cannot be changed
        # interval: degree, not rad
        suffix = str(suffix)
        if suffix != '':
            suffix = '_' + suffix

        #img_cnt = math.ceil(360/interval)
        names = ['F', 'R', 'B', 'L']  # forward, backward, left, right
        #interval = math.ceil(360 / len(names))
        for idx, name in enumerate(names):
            yaw = road_compassA + idx * 90
            # print('idx: ', idx)
            # print('name:', name)
            # print('yaw: ', yaw)
            self.getImagefrmAngle(lon, lat, saved_path, prefix, name + suffix, width, height, pitch, yaw)

    def getImage8DirectionfrmLonlat(self, lon: float, lat: float, saved_path='Panos', prefix='', suffix='', width=1024, height=768, pitch=0, road_compassA=0):
        # w maximum: 1024
        # h maximum: 768
        # FOV should be 90, cannot be changed
        # interval: degree, not rad
        suffix = str(suffix)
        if suffix != '':
            suffix = '_' + suffix

        #img_cnt = math.ceil(360/interval)
        #names = ['F', 'R', 'B', 'L']  # forward, backward, left, right
        names = ['F', 'FR', 'R', 'RB', 'B', 'BL', 'L', 'LF']  # forward, backward, left, right
        interval = math.ceil(360 / len(names))
        for idx, name in enumerate(names):
            yaw = road_compassA + idx * interval
            # print('idx: ', idx)
            # print('name:', name)
            # print('yaw: ', yaw)
            self.getImagefrmAngle(lon, lat, saved_path, prefix, name + suffix, width, height, pitch, yaw)

    def getImage4DirectionfrmLonlats(self, list_lonlat, saved_path='Panos', prefix='', suffix='', width=1024, height=768, pitch=0, road_compassA=0):
        #print(len(list_lonlat))


        start_time = time.time()
        Cnt = 0
        Cnt_interval = 100
        origin_len = len(list_lonlat)

        while len(list_lonlat) > 0:
            try:
                #print(list_lonlat.pop(0))
                lon, lat, id, prefix, road_compassA = list_lonlat.pop(0)
                prefix = str(id)

                print('Current row :', id)
                self.getImage8DirectionfrmLonlat(lon, lat, saved_path, prefix, suffix, width, height, pitch, road_compassA)
                current_len = len(list_lonlat)
                Cnt = origin_len - current_len
                if Cnt % Cnt_interval == (Cnt_interval - 1):
                    print(
                        "Prcessed {} / {} items. Processing speed: {} points / hour.".format(Cnt, origin_len, int(Cnt / (time.time() - start_time + 0.001) * 3600)))
            except Exception as e:
                print("Error in getImage4DirectionfrmLonlats(): ", e, id)
                current_len = len(list_lonlat)
                continue

    def getImage4DirectionfrmLonlats_mp(self, list_lonlat_mp, saved_path='Panos', Process_cnt = 6, prefix='', suffix='', width=1024, height=768, pitch=0, road_compassA=0):
        #statuses = []      # succeeded: 1; failed: 0
        pool = mp.Pool(processes=Process_cnt)

        for i in range(Process_cnt):
            pool.apply_async(self.getImage4DirectionfrmLonlats, args=(list_lonlat_mp, saved_path, prefix, suffix, width, height, pitch, road_compassA))
        pool.close()
        pool.join()



class GSV_depthmap(object):

    def getPanoIdDepthmapfrmLonlat(self, lon, lat, dm=1, saved_path='', prefix='', suffix=''):
        url = f''
        r = requests.get('url')
        print(r)

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
        return transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), lon, lat)

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
            theta = (h - y - 0.5) / h * np.pi
            sin_theta[y] = np.sin(theta)
            cos_theta[y] = np.cos(theta)

        for x in range(w):
            phi = (w - x - 0.5) / w * 2 * np.pi + np.pi / 2
            sin_phi[x] = np.sin(phi)
            cos_phi[x] = np.cos(phi)

        for y in range(h):
            for x in range(w):
                planeIdx = indices[y * w + x]

                v[0] = sin_theta[y] * cos_phi[x]
                v[1] = sin_theta[y] * sin_phi[x]
                v[2] = cos_theta[y]

                if planeIdx > 0:
                    plane = planes[planeIdx]
                    t = np.abs(
                        plane["d"]
                        / (
                                v[0] * plane["n"][0]
                                + v[1] * plane["n"][1]
                                + v[2] * plane["n"][2]
                        )
                    )
                    depthMap[y * w + (w - x - 1)] = t
                else:
                    depthMap[y * w + (w - x - 1)] = 9999
        return {"width": w, "height": h, "depthMap": depthMap}

    #


if __name__ == '__main__':

    print("Started to test...")
    gpano = GPano()

    #### Test for getPanoIDfrmLonlat()
    #print(gpano.getPanoIDfrmLonlat(-74.24756, 40.689524))  # Works well.

    # Using multi_processing to download panorama images from a list
    #list_lonlat = pd.read_csv(r'Morris_county\Morris_10m_points.csv')
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
#------------------------------------


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

    print("Started to lonlat2WebMercator().")
    list_lonlat = pd.read_csv(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\2015_Street_Tree_Census_-_Tree_Data.csv')
    print("Got rows of :", len(list_lonlat))
    list_lonlat = list_lonlat[:10]
    for idx, row in list_lonlat.iterrows():
        print(row.longitude, row.latitude)
        print(gpano.lonlat2WebMercator(row.longitude, row.latitude))

#------------------------------------


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
    depthMapData = parser.parse(s)
    # parse first bytes to describe data
    header = parser.parseHeader(depthMapData)
    # parse bytes into planes of float values
    data = parser.parsePlanes(header, depthMapData)
    # compute position and values of pixels
    depthMap = parser.computeDepthMap(header, data["indices"], data["planes"])
    # process float 1D array into int 2D array with 255 values
    im = depthMap["depthMap"]
    #print(im)
    im[np.where(im == max(im))[0]] = 9999
    # if min(im) < 0:
    #     im[np.where(im < 0)[0]] = 0
    im = im.reshape((depthMap["height"], depthMap["width"]))#.astype(int)
    # display image
    img = PIL.Image.fromarray(im)
    img.save(r'D:\Code\StreetView\Essex\t\img.tiff')
    plt.imshow(im)
    plt.show()
    print(im)
    #plt.imsave(r'D:\Code\StreetView\Essex\t\img.tiff', im)
    #------------------------------------

    print(features.check('libtiff'))
    print(features.check('libtiff'))