import GPano
import R_tree_utils
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
import multiprocessing as mp

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import shapely.wkt
import os
import fiona

from rtree import index
import shapely

import logging
import sys
import pprint

from pyproj import Proj, transform
from geopy.distance import geodesic

gpano = GPano.GPano()
gsv = GPano.GSV_depthmap()

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
        # logger.info(os.path.basename(file))

logger = logging.getLogger('Log.file')

class shoot_polygons():
    def __init__(self, polygons: list):  # list of shapely.geometry.Polygon
        self.polygons = polygons
        self.rtree = index.Index()

        for idx, polygon in enumerate(self.polygons):
            self.rtree.insert(idx, polygon.bounds)




def shoot_houston_building():
    try:
        shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\building_in_flood_houson.shp'
        saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\street_images'


        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        buildings = fiona.open(shape_file)

        outEPSG = 'epsg:4326'
        inEPSG = 'epsg:3857'

        # generate shaple.Polygons.
        shoot_ply = shoot_polygons([])
        for idx, building in enumerate(buildings):
            try:
                geometry = building['geometry']['coordinates']
                if len(geometry) > 1:
                    logger.info('Polygon # %s have multiple (%d) parts.', idx, len(geometry))
                    geometry = geometry[:1]
                geometry = np.array(geometry).squeeze(0)

                inProj = Proj(inEPSG)  # webmap mercator
                outProj = Proj(outEPSG)

                geometry = transform(inProj, outProj, geometry[:, 0], geometry[:, 1])  # return lat, lon
                geometry = Polygon(zip(geometry[1], geometry[0]))

                # insert to r-tree
                bound = geometry.bounds
                shoot_ply.rtree.insert(idx, bound)
                shoot_ply.polygons.append(geometry)
            except Exception as e:
                logger.error("Error in building polygons: %s", e)
                continue

        for idx, polygon in enumerate(shoot_ply.polygons):
            try:
                x, y = polygon.centroid
            except Exception as e:
                logger.error("Error in enumerate polygons: %s", e)
                continue


    except Exception as e:
        logger.error("shoot_houston_building: %s", e)


    # building r-tree
    # r_tree = R_tree_utils.rtree_utils(shape_file.replace(".shp", '')).r_idx
    # for idx, building in enumerate(buildings):
    #     geometry = building['geometry']['coordinates']
    #     geometry = np.array(geometry).squeeze(0)
    #
    #     inProj = Proj('epsg:3857')  # webmap mercator
    #     outProj = Proj('epsg:4326')
    #
    #     geometry = transform(inProj, outProj, geometry[:, 0], geometry[:, 1])  # return lat, lon
    #     geometry = Polygon(zip(geometry[1], geometry[0]))
    #
    #     # insert to r-tree
    #     bound = geometry.bounds
    #     r_tree.insert(idx, bound)
    #
    # if not os.path.exists(saved_path):
    #     os.mkdir(saved_path)
    #
    #
    # for i in tqdm(range(len(buildings))):
    #     # FID, area_m, ACCTID, story, GEOID, tract_pop,lon, lat, geometry = buildings.iloc[i]
    #     # geometry = shapely.wkt.loads(geometry)
    #
    #     try:
    #         geometry = buildings[i]['geometry']['coordinates']
    #         print("Processing (FID): ", i)
    #         geometry = np.array(geometry).squeeze(0)
    #
    #         x = geometry[:, 0].mean()
    #         y = geometry[:, 1].mean()
    #
    #         inProj = Proj('epsg:3857')  # NAD83 / Massachusetts Mainland (ftUS)
    #         outProj = Proj('epsg:4326')
    #
    #         # lon, lat = transform(inProj, outProj, x, y)
    #
    #         geometry = transform(inProj, outProj, geometry[:, 0], geometry[:, 1])  # return lat, lon
    #
    #         geometry = Polygon(zip(geometry[1], geometry[0]))
    #
    #         lat, lon = geometry.centroid.y, geometry.centroid.x
    #
    #         distance_threshold = 50  # meter
    #
    #         ret = gpano.shootLonlat(lon, lat, polygon=geometry, saved_path=saved_path, prefix='', width=576, height=768,
    #                                 fov=90, distance_threshold=distance_threshold, r_tree=r_tree)
    #
    #         # shootLonlat(self, ori_lon, ori_lat, saved_path='', views=1, prefix='', suffix='', width=1024,
    #         #                height=768, pitch=0):
    #     except Exception as e:
    #         print("Error in download_buildings_boston(): ", e, i)

if __name__ == "__main__":
    shoot_houston_building()