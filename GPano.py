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
WINDOWS_SIZE = '100, 100'
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
Loading_time = 5
web_driver_path = r'chromedriver.exe'
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
    def getPanoJPGsfrmLonlats(self, list_lonlat, saved_path, zoom=4):
        """ Obtain panomara images from a list of lat/lon: [(lon, lat), ...]

        """
        statuses = []      # succeeded: 1; failed: 0
        #print(list_lonlat.pop(0))

        while len(list_lonlat) > 0:
            lon, lat = list_lonlat.pop(0)
            print(lon, lat)
            self.getPanoJPGfrmLonlat(lon, lat, saved_path)
        return statuses

    def getPanoJPGfrmLonlat(self, lon: float, lat: float, saved_path: str, zoom: int = 4) -> bool:
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

        if adcode[0] == 0:
            print(adcode[0], "is not a PanoID.")
            return


        print(adcode)  # Works well.
        for x in range(30):  # test for the size of map
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

        for x in range(30):
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
            mapname = os.path.join(saved_path, (adcode[1] + '_' + adcode[2] + '_' + adcode[0] + '.jpg'))
            target.save(mapname)
        except Exception as e:
            print("Error in getPanoJPGfrmLonlat():", e)


        return status


    def getPanosfrmLonlats_mp(self, list_lonlat_mp, saved_path, zoom=4, Process_cnt=4):
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


def getPanosfrmLonlats_mp(list_lonlat_mp, saved_path, zoom=4, Process_cnt=4):
    """ Multi_processing version of getPanosfrmLonlats()
        Obtain panomara images from a list of lat/lon: [(lon, lat), ...]

    """
    statuses = []      # succeeded: 1; failed: 0
    gpano2 = GPano()
    print(gpano2)
    pool = mp.Pool(processes=Process_cnt)
    for i in range(Process_cnt):
        pool.apply_async(gpano2.getPanoJPGsfrmLonlats, args=(list_lonlat_mp, saved_path))
    pool.close()
    pool.join()


if __name__ == '__main__':
    gpano = GPano()

    # Test getPanoIDfrmLonlat()
    #print(gpano.getPanoIDfrmLonlat(-74.24756, 40.689524))  # Works well.

    # Using multi_processing to download panorama images from a list
    list_lonlat = pd.read_csv(r'Morris_county\Morris_10m_points.csv')
    list_lonlat = list_lonlat[40000:40200]
    mp_lonlat = mp.Manager().list()
    for idx, row in list_lonlat.iterrows():
        mp_lonlat.append([row['lon'], row['lat']])
        #gpano.getPanoJPGfrmLonlat(row['lon'], row['lat'], saved_path='jpg')
    print(mp_lonlat)
    gpano.getPanosfrmLonlats_mp(mp_lonlat, saved_path='jpg', Process_cnt=4)





