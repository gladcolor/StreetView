"""
Designed by Huan Ning, gladcolor@gmail.com, 2019.09.04

"""

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
import csv
import math
import sys
WINDOWS_SIZE = '100, 100'
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
Loading_time = 5
web_driver_path = r'K:\Research\StreetView\Google_street_view\chromedriver.exe'
driver = webdriver.Chrome(executable_path=web_driver_path, chrome_options=chrome_options)
#Process_cnt = 10

"""
Read Me 
lon/lat = longitude/latitude
The current objective of GPano class is to download the panorama image from Google Street View according to lon/lat.
Main workflow: get panorama image ID based on lon/lat -> according to the panorama id, get tiles of a Panorama image and then mosaic them 
Please implement all the methods. I have written some tips (not code) in the method body to assist you. -- Huan
"""

class GPano:
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
    def getPanoIDfrmLonlat(self, lon:float, lat:float,) -> (str, float, float):
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
            print('prefix:', prefix)
            prefix = prefix + '_'
        if suffix != "":
            suffix = '_' + suffix


        try:
            response = requests.get(url1)
            image = Image.open(BytesIO(response.content))

            jpg_name = os.path.join(saved_path, (prefix + str(lon) + '_' + str(lat) + '_' + str(pitch) + str(int(yaw)) + suffix + '.jpg'))
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

    def getImage4DirectionfrmLonlats(self, list_lonlat, saved_path='Panos', prefix='', suffix='', width=1024, height=768, pitch=0, road_compassA=0):
        #print(len(list_lonlat))


        start_time = time.time()
        Cnt = 0
        Cnt_interval = 100
        origin_len = len(list_lonlat)
        current_len = origin_len
        while current_len > 0:
            try:
                #print(list_lonlat.pop(0))
                lon, lat, id, road_compassA = list_lonlat.pop(0)
                prefix = str(id)
                current_len = len(list_lonlat)
                print('id :', id)
                self.getImage4DirectionfrmLonlat(lon, lat, saved_path, prefix, suffix, width, height, pitch, road_compassA)
                Cnt = current_len - origin_len
                if Cnt % Cnt_interval == (Cnt_interval - 1):
                    print(
                        "Process speed: {} points / hour.".format(int(Cnt / (time.time() - start_time + 0.001) * 3600)))
            except Exception as e:
                print("Error in getImage4DirectionfrmLonlats(): ", e, id)
                continue

    def getImage4DirectionfrmLonlats_mp(self, list_lonlat_mp, saved_path='Panos', Process_cnt = 6, prefix='', suffix='', width=1024, height=768, pitch=0, road_compassA=0):
        #statuses = []      # succeeded: 1; failed: 0
        pool = mp.Pool(processes=Process_cnt)

        for i in range(Process_cnt):
            pool.apply_async(self.getImage4DirectionfrmLonlats, args=(list_lonlat_mp, saved_path, prefix, suffix, width, height, pitch, road_compassA))
        pool.close()
        pool.join()


if __name__ == '__main__':
    gpano = GPano()

    # Test getPanoIDfrmLonlat()
    #print(gpano.getPanoIDfrmLonlat(-74.24756, 40.689524))  # Works well.

    # Using multi_processing to download panorama images from a list
    #list_lonlat = pd.read_csv(r'Morris_county\Morris_10m_points.csv')
    print(sys.getfilesystemencoding())
    print(sys.getdefaultencoding())
    #list_lonlat = pd.read_csv(r'G:\My Drive\Research\sidewalk\test_data.csv', quoting=csv.QUOTE_ALL, engine="python", encoding='utf-8')
    list_lonlat = pd.read_csv(r'Morris_county\Morris_10m_points.csv')
    list_lonlat = list_lonlat[:1000]
    list_lonlat = list_lonlat.fillna(0)
    mp_lonlat = mp.Manager().list()
    #print(len(list_lonlat))
    for idx, row in list_lonlat.iterrows():
        mp_lonlat.append([row['lon'], row['lat'], int(idx + 1), row['CompassA']])
        #gpano.getPanoJPGfrmLonlat(row['lon'], row['lat'], saved_path='jpg')
        #print(idx)
        #gpano.getImage4DirectionfrmLonlat(row['lon'], row['lat'], saved_path=r'G:\My Drive\Sidewalk_extraction\Morris_jpg', road_compassA=row['CompassA'], prefix=int(row['id']))
    #print(mp_lonlat)
    #gpano.getPanosfrmLonlats_mp(mp_lonlat, saved_path=r'G:\My Drive\Sidewalk_extraction\Morris_jpg', Process_cnt=1)
    gpano.getImage4DirectionfrmLonlats_mp(mp_lonlat, saved_path=r'G:\My Drive\Sidewalk_extraction\Morris_jpg', Process_cnt=4)





