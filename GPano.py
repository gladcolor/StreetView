"""
Designed by Huan Ning, gladcolor@gmail.com, 2019.09.04

"""

import multiprocessing as mp
import selenium

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
WINDOWS_SIZE = '100, 100'
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--windows-size=%s" % WINDOWS_SIZE)
Loading_time = 5
web_driver_path = r'chromedriver.exe'
driver = webdriver.Chrome(executable_path=web_driver_path, chrome_options=chrome_options)
Process_cnt = 10

class GPano:
    # Obtain a panomaro image from Google Street View Map
    def getPanosfrmLonlats(self, list_lonlat, saved_path, zoom=4):
        """ Obtain panomara images from a list of lat/lon: [(lon, lat), ...]

        """
        statuses = []      # succeeded: 1; failed: 0
        for lon, lat in list_lonlat.pop(0):



        return statuses

    def getPanoJPGfrmLonlat(self, lon: float, lat: float, saved_path: str, zoom: int = 4) -> bool:
        status = 0
        PanoID, lon_pano, lat_pano = getPanoIDfrmLonlat(lat, lon, zoom=4)

        return status



    # Obtain a panomara_ID according to lon/lat
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
        url = f"https://www.google.com/maps/@{lat},{lon},3a,{fov}y,{heading}h,{tilt}t/data={url_part1}{url_part2}"

        #print(url)
        try:
            driver.get(url)
            time.sleep(4)
            new_url = driver.current_url
            if new_url == url:
                time.sleep(2)
            if new_url == url:
                PanoID = 0
                lon_pano = 0
                lat_pano = 0
            else:
                lat_pano, lon_pano = new_url.split(',')[:2]
                lat_pano = lat_pano.split("@")[1]
                pos1 = new_url.find(url_part1)
                pos2 = new_url.find(url_part2)
                PanoID = new_url[(pos1 + len(url_part1)):pos2]
                #print(new_url)

            return PanoID, lon_pano, lat_pano   # if cannot find the panomara, return (0, 0, 0)
        except Exception as e:
            print("Error in getPanoIDfrmLonlat()", e)

    def getPanoTiles(selft, PanoID):
        imgs = []
        return imgs, lat_acc, lon_acc

if __name__ == '__main__':
    gpano = GPano()

    # Test getPanoIDfrmLonlat()
    print(gpano.getPanoIDfrmLonlat(-74.24756, 40.689524))  # Works well.

    # Using multi_processing to download panorama images from a list
    list_lonlat = pd.read_csv(r'morris_points_10.csv')
    list_lonlat = list_lonlat[:200]
    mp_lonlat = mp.Manager().list()
    for idx, row in list_lonlat.iterrows():
        mp_lonlat.append([row['lon'], row['lat']])




