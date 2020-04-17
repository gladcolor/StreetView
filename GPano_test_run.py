from GPano import *
import GPano
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import  sqlite3
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import shapely.wkt
import os


gpano = GPano.GPano()
gsv = GPano.GSV_depthmap()





def test_getPanoJPGfrmArea():
    print('started! ')

    # csv file needs POINT_X,POINT_Y.
    pts = gpano.readRoadSeedsPts_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\House\maryland\Maryland_Road_Centerlines__Comprehensive\Maryland_Road_Centerlines_pts.csv')
    # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
    #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
    coords = gpano.readCoords_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\House\maryland\maryland_bou.csv')
    polygon = gpano.formPolygon(coords)
    saved_path = r'J:\Maryland\jsons2'
    random.shuffle(pts)

    # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
    gpano.getPanoJPGfrmArea_mp('json_only', pts, saved_path, coords, zoom=4, Process_cnt=8)

def test_getPanoJPGfrmArea_west_philly():
    print('started! ')
    pts = gpano.readRoadSeedsPts_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\vectors\west_philly\west_road_pts.cvs')
    # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
    #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
    coords = gpano.readCoords_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\vectors\west_philly\boundary.csv')
    polygon = gpano.formPolygon(coords)
    saved_path = r'J:\Research\Trees\west_philly\street_images'
    random.shuffle(pts)

    # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
    gpano.getPanoJPGfrmArea_mp([90, 270], pts,  saved_path, coords, fov=120, Process_cnt=10)
    # remember to change fov=90 as default.
    # getImagefrmAngle(self, lon: float, lat: float, saved_path='', prefix='', suffix='', width=1024, height=768,
    #                          pitch=0, yaw=0, fov=120): #


def test_getPanoJPGfrmArea_philly():
    print('started! ')
    pts = gpano.readRoadSeedsPts_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\Philly_road_pts.csv')
    # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
    #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
    coords = gpano.readCoords_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\Philly__boundary.csv')
    polygon = gpano.formPolygon(coords)
    saved_path = r'X:\Shared drives\Group_research\Trunk_research\Datasets\Pilly\street_view'
    random.shuffle(pts)

    # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
    gpano.getPanoJPGfrmArea_mp([60, 90, 120, 240, 270, 300], pts,  saved_path, coords, Process_cnt=10)

def download_buildings():
    buildings = pd.read_csv(r"K:\OneDrive_NJIT\OneDrive - NJIT\Research\House\maryland\footprints\building_attri2.csv", sep=',')

    buildings = buildings.iloc[(147337+50370+329436+201628+139335+331656):]

    os.system("rm -rf I:\\Research\\House\\images\\*")
    os.system("mkdir I:\\Research\\House\\images\\1.0 I:\\Research\\House\\images\\1.5 I:\\Research\\House\\images\\2.0 I:\\Research\\House\\images\\2.5 I:\\Research\\House\\images\\3.0 I:\\Research\\House\\images\\3.5 I:\\Research\\House\\images\\4.0 I:\\Research\\House\\images\\4.5")

    # Process_cnt = 10
    # pool = mp.Pool(processes=Process_cnt)https://m.tsemporium.com/zh_cn/
    #
    # for i in range(Process_cnt):
    #     pool.apply_async(self.getPanoJPGfrmArea, args=(yaw_list, seed_pts_mp, saved_path, boundary_vert, zoom))
    # pool.close()
    # pool.join()


    for i in tqdm(range(len(buildings))):
        FID, area_m, ACCTID, story, GEOID, tract_pop,lon, lat, geometry = buildings.iloc[i]
        geometry = shapely.wkt.loads(geometry)
        print("Processing (FID): ", FID)
        try:
            ret = gpano.shootLonlat(lon, lat, polygon=geometry, saved_path=f'I:\\Research\\House\\images\\{story}', prefix=ACCTID, width=576, height=768,  fov=90)

             # shootLonlat(self, ori_lon, ori_lat, saved_path='', views=1, prefix='', suffix='', width=1024,
             #                height=768, pitch=0):
        except Exception as e:
            print("Error in download_buildings(): ", e, FID)



if __name__ == '__main__':
    # test_getPanoJPGfrmArea()
    # download_buildings()
    # test_getPanoJPGfrmArea_philly()
    test_getPanoJPGfrmArea_west_philly()

