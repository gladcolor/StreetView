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

import pyproj
from pyproj import Proj, transform, Transformer
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


class shoot_polygons():
    def __init__(self, polygons: list):  # list of shapely.geometry.Polygon
        self.polygons = polygons
        self.rtree = index.Index()

        for idx, polygon in enumerate(self.polygons):
            self.rtree.insert(idx, polygon.bounds)

def create_rtree(shape_file: str, inEPSG='EPSG:4326', outEPSG='EPSG:4326') :
    try:
        buildings = fiona.open(shape_file)

        # outEPSG = 'EPSG:4326'
        # inEPSG = 'EPSG:3857'
        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # rtree
        rtree_path = shape_file.replace(".shp", '_rtree')
        p = index.Property()
        p.overwrite = True
        r_tree = index.Index(rtree_path, properties=p)

        print_interval = int(len(buildings) / 1000)

        for idx, building in tqdm(enumerate(buildings[:])):
            try:
                if idx % print_interval == 0:
                    logger.info("Processing polyogn #: %d", idx)

                bound = fiona.bounds(building)
                # logger.info("bound: %s", bound)
                bound = list(bound)
                # logger.info("bound: %s", bound)

                bound[0], bound[1] = transformer.transform(bound[0], bound[1])
                # logger.info("bound: %s", bound)

                bound[2], bound[3] = transformer.transform(bound[2], bound[3])

                r_tree.insert(idx, bound)

            except Exception as e:
                logger.error("Error in building polygons: %s", e)
                continue
        r_tree.close()

    except Exception as e:
        logger.error("Error in creating rtree: %s", e)

def fionaPolygon2shaple(fiona_geometry) ->shapely.geometry.Polygon:  # only consider the exterior ring
    coords = fiona_geometry['coordinates']
    if len(coords) > 1:
        coords = coords[:1]
    coords = np.array(coords).squeeze(0)
    return shapely.geometry.Polygon(zip(coords[:, 0], coords[:, 1]))

def getShooting_triangle(viewpoint: tuple, polygon: shapely.geometry.Polygon) -> np.array:
    points_list = np.array(polygon.exterior.coords)
    viewpoint = np.array(viewpoint)
    # get nearest two vertices
    delta = points_list - viewpoint

    delta_squre = delta[:, 0] ** 2 + delta[:, 1] ** 2
    roots = delta_squre ** 0.5
    sorted_idx = np.argsort(roots[:-1])[:2]
    point1 = points_list[sorted_idx[0]]
    point2 = points_list[sorted_idx[1]]
    point_mid = (point1 + point2)/2
    heading = gpano.getDegreeOfTwoLonlat(viewpoint[1], viewpoint[0], point_mid[1], point_mid[0])   # notice: (lat, lon, lat, lon)
    triangle = Polygon([viewpoint, point1, point2, viewpoint])
    return triangle, heading

def shapelyReproject(transformer: pyproj.transformer, polygon: shapely.geometry.Polygon) -> shapely.geometry.Polygon:
    # consider only exterior
    coords = polygon.exterior.coords.xy
    xs, ys = transformer.transform(coords[0], coords[1])

    return shapely.geometry.Polygon(zip(xs, ys))

def shoot_houston_building():
    try:
        shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\building_in_flood_houson.shp'
        saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\street_images'

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "_info.log"))

        inEPSG = 'EPSG:3857'
        outEPSG = 'EPSG:4326'

        rtree_path = shape_file.replace(".shp", '_rtree.idx')
        r_tree = None

        w = 768
        h = 768

        if os.path.exists(rtree_path):
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))
            logger.info("Loading the Rtree: %s", rtree_path)
        else:
            logger.info("Creating the Rtree: %s", rtree_path)
            create_rtree(shape_file, inEPSG=inEPSG, outEPSG=outEPSG)

            logger.info("Loading the Rtree: %s", rtree_path)
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))

        # test = r_tree.intersection((-95.608977, 29.736570, -95.408977, 29.936570))

        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        buildings = fiona.open(shape_file)



        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # generate shaple.Polygons.
        shoot_ply = shoot_polygons([])
        start = 70000
        for idx  in tqdm(range(start, len(buildings))):
            try:
                building = buildings[idx]
                logger.info("Processing polyogn #: %d", idx)
                geometry = building['geometry']['coordinates']
                ID = str(building['properties']['ID'])
                if len(geometry) > 1:
                    logger.info('Polygon # %s have multiple (%d) parts.', idx, len(geometry))
                    geometry = geometry[:1]
                geometry = np.array(geometry).squeeze(0)

                xs, ys = transformer.transform(geometry[:, 0], geometry[0:, 1])

                polygon = Polygon(zip(xs, ys))

                x, y = polygon.centroid.xy  # x is an array, the number is x[0]
                x = x[0]
                y = y[0]

                logger.info("polygon.centroid: %f, %f", x, y)

                panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)

                if panoId == 0:
                    logger.info("Cannot find a street view image at : %s, %s ", x, y)
                    continue

                viewpoint = np.array((lon, lat))
                # triangle = getShooting_triangle(viewpoint, polygon)

                min_rotated_rectangle  = polygon.minimum_rotated_rectangle
                # points_list = min_rotated_rectangle.exterior.coords

                triangle, heading = getShooting_triangle(viewpoint, min_rotated_rectangle)
                GSV_url = gpano.getGSV_url_frm_lonlat(lon, lat, heading)
                logger.info("GSV url: %s", GSV_url)


                # find intersects in the r-tree
                bound = triangle.bounds
                intersects = r_tree.intersection(bound)
                intersects = list(intersects)

                isIntersected = False
                for inter in intersects:
                    if inter == idx:
                        continue
                    building = buildings[inter]['geometry']
                    building = fionaPolygon2shaple(building)
                    building = shapelyReproject(transformer, building)
                    isIntersected = triangle.intersects(building)
                    if isIntersected:
                        logger.info("Occluded by other houses.")
                        break

                if isIntersected:
                    # logger.info("Occluded by other houses.")
                    continue

                ret = gpano.shootLonlat(lon, lat, polygon=min_rotated_rectangle, saved_path=saved_path, prefix=ID, width=w,
                                        height=h, fov=90)
                # logger.info("Google Street View: %s", gpano.getGSV_url_frm_lonlat(lon, lat, ))
                
                # logger.info("intersects: %s", intersects)

            except Exception as e:
                logger.error("Error in building polygons: %s", e)
                continue




    except Exception as e:
        logger.error("shoot_houston_building: %s", e)


def shoot_philly_building():
    try:
        shape_file = r'J:\Research\Resilience\data\building_in_floodzone.shp'
        saved_path = r'J:\Research\Resilience\data\street_images'

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "_info.log"))

        inEPSG = 'EPSG:4326'
        outEPSG = 'EPSG:4326'

        rtree_path = shape_file.replace(".shp", '_rtree.idx')
        r_tree = None

        w = 768
        h = 1024

        if os.path.exists(rtree_path):
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))
            logger.info("Loading the Rtree: %s", rtree_path)
        else:
            logger.info("Creating the Rtree: %s", rtree_path)
            create_rtree(shape_file, inEPSG=inEPSG, outEPSG=outEPSG)

            logger.info("Loading the Rtree: %s", rtree_path)
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))

        # test = r_tree.intersection((-95.608977, 29.736570, -95.408977, 29.936570))

        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        buildings = fiona.open(shape_file)

        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # generate shaple.Polygons.
        shoot_ply = shoot_polygons([])
        start = 0
        for idx in tqdm(range(start, len(buildings))):
            try:
                building = buildings[idx]
                logger.info("Processing polyogn #: %d", idx)
                geometry = building['geometry']['coordinates']
                ID = str(building['properties']['ID'])
                if len(geometry) > 1:
                    logger.info('Polygon # %s have multiple (%d) parts.', idx, len(geometry))
                    geometry = geometry[:1]
                geometry = np.array(geometry).squeeze(0)

                xs, ys = transformer.transform(geometry[:, 0], geometry[0:, 1])

                polygon = Polygon(zip(xs, ys))

                x, y = polygon.centroid.xy  # x is an array, the number is x[0]
                x = x[0]
                y = y[0]

                logger.info("polygon.centroid: %f, %f", x, y)

                panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)

                if panoId == 0:
                    logger.info("Cannot find a street view image at : %s, %s ", x, y)
                    continue

                viewpoint = np.array((lon, lat))
                # triangle = getShooting_triangle(viewpoint, polygon)

                min_rotated_rectangle = polygon.minimum_rotated_rectangle
                # points_list = min_rotated_rectangle.exterior.coords

                triangle, heading = getShooting_triangle(viewpoint, min_rotated_rectangle)
                GSV_url = gpano.getGSV_url_frm_lonlat(lon, lat, heading)
                logger.info("GSV url: %s", GSV_url)

                # find intersects in the r-tree
                bound = triangle.bounds
                intersects = r_tree.intersection(bound)
                intersects = list(intersects)

                isIntersected = False
                for inter in intersects:
                    if inter == idx:
                        continue
                    building = buildings[inter]['geometry']
                    building = fionaPolygon2shaple(building)
                    building = shapelyReproject(transformer, building)
                    isIntersected = triangle.intersects(building)
                    if isIntersected:
                        logger.info("Occluded by other houses.")
                        break

                if isIntersected:
                    # logger.info("Occluded by other houses.")
                    continue

                ret = gpano.shootLonlat(lon, lat, polygon=min_rotated_rectangle, saved_path=saved_path, prefix=ID,
                                        width=w,
                                        height=h, fov=90)
                # logger.info("Google Street View: %s", gpano.getGSV_url_frm_lonlat(lon, lat, ))

                # logger.info("intersects: %s", intersects)

            except Exception as e:
                logger.error("Error in building polygons: %s", e)
                continue




    except Exception as e:
        logger.error("shoot_philly_building: %s", e)



def shoot_oceancity_building():
    try:
        shape_file = r'J:\Research\Resilience\data\OcenaCity\building_shp\OceanCity_buildings_floodzone.shp'
        saved_path = r'J:\Research\Resilience\data\OcenaCity\images'

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "_info.log"))

        inEPSG = 'EPSG:4326'
        outEPSG = 'EPSG:4326'

        rtree_path = shape_file.replace(".shp", '_rtree.idx')
        r_tree = None

        w = 768
        h = 768

        if os.path.exists(rtree_path):
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))
            logger.info("Loading the Rtree: %s", rtree_path)
        else:
            logger.info("Creating the Rtree: %s", rtree_path)
            create_rtree(shape_file, inEPSG=inEPSG, outEPSG=outEPSG)

            logger.info("Loading the Rtree: %s", rtree_path)
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))

        # test = r_tree.intersection((-95.608977, 29.736570, -95.408977, 29.936570))

        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        buildings = fiona.open(shape_file)

        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # generate shaple.Polygons.
        shoot_ply = shoot_polygons([])
        start = 456
        for idx in tqdm(range(start, len(buildings))):
            try:
                building = buildings[idx]
                logger.info("Processing polyogn #: %d", idx)
                geometry = building['geometry']['coordinates']
                ID = str(building['properties']['ID'])
                if len(geometry) > 1:
                    logger.info('Polygon # %s have multiple (%d) parts.', idx, len(geometry))
                    geometry = geometry[:1]
                geometry = np.array(geometry).squeeze(0)

                xs, ys = transformer.transform(geometry[:, 0], geometry[0:, 1])

                polygon = Polygon(zip(xs, ys))

                x, y = polygon.centroid.xy  # x is an array, the number is x[0]
                x = x[0]
                y = y[0]

                logger.info("polygon.centroid: %f, %f", x, y)

                panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)

                if panoId == 0:
                    logger.info("Cannot find a street view image at : %s, %s ", x, y)
                    continue

                viewpoint = np.array((lon, lat))
                # triangle = getShooting_triangle(viewpoint, polygon)

                min_rotated_rectangle = polygon.minimum_rotated_rectangle
                # points_list = min_rotated_rectangle.exterior.coords

                triangle, heading = getShooting_triangle(viewpoint, min_rotated_rectangle)
                GSV_url = gpano.getGSV_url_frm_lonlat(lon, lat, heading)
                logger.info("GSV url: %s", GSV_url)

                # find intersects in the r-tree
                bound = triangle.bounds
                intersects = r_tree.intersection(bound)
                intersects = list(intersects)

                isIntersected = False
                for inter in intersects:
                    if inter == idx:
                        continue
                    building = buildings[inter]['geometry']
                    building = fionaPolygon2shaple(building)
                    building = shapelyReproject(transformer, building)
                    isIntersected = triangle.intersects(building)
                    if isIntersected:
                        logger.info("Occluded by other houses.")
                        break

                if isIntersected:
                    # logger.info("Occluded by other houses.")
                    continue

                ret = gpano.shootLonlat(lon, lat, polygon=min_rotated_rectangle, saved_path=saved_path, prefix=ID,
                                        width=w,
                                        height=h, fov=90)
                # logger.info("Google Street View: %s", gpano.getGSV_url_frm_lonlat(lon, lat, ))

                # logger.info("intersects: %s", intersects)

            except Exception as e:
                logger.error("Error in building polygons: %s", e)
                continue




    except Exception as e:
        logger.error("shoot_philly_building: %s", e)

def shoot_Galveston_buildings():
    try:
        shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\Galveston\buildings_WGS84.shp'
        saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\Galveston\street_images_square'

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "info.log"))
        w = 768
        h = 768
        # logger.info(os.path.basename(file))

        logger = logging.getLogger('console_only')


        rtree_path = shape_file.replace(".shp", '_rtree.idx')
        r_tree = None

        if os.path.exists(rtree_path):
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))
            logger.info("Loading the Rtree: %s", rtree_path)
        else:
            logger.info("Creating the Rtree: %s", rtree_path)
            create_rtree(shape_file)

            logger.info("Loading the Rtree: %s", rtree_path)
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))


        # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        buildings = fiona.open(shape_file)

        inEPSG  = 'EPSG:4326'
        outEPSG = 'EPSG:4326'

        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # generate shaple.Polygons.
        shoot_ply = shoot_polygons([])
        start = 0
        for idx in tqdm(range(start, len(buildings))):
            try:
                building = buildings[idx]
                logger.info("Processing polyogn #: %d", idx)
                geometry = building['geometry']['coordinates']

                row_id = building['properties']['ID']
                row_id = int(row_id)

                if len(geometry) > 1:
                    logger.info('Polygon # %s have multiple (%d) parts.', idx, len(geometry))
                    geometry = geometry[:1]
                geometry = np.array(geometry).squeeze(0)

                # coords = polygon.exterior.coords.xy  # coords:

                xs, ys = transformer.transform(geometry[:, 0], geometry[0:, 1])

                polygon = Polygon(zip(xs, ys))

                x, y = polygon.centroid.xy  # x is an array, the number is x[0]
                x = x[0]
                y = y[0]

                logger.info("polygon.centroid: %f, %f", x, y)

                panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)

                if panoId == 0:
                    logger.info("Cannot find a street view image at : %s, %s ", x, y)
                    continue

                viewpoint = np.array((lon, lat))
                # triangle = getShooting_triangle(viewpoint, polygon)

                min_rotated_rectangle = polygon.minimum_rotated_rectangle
                # points_list = min_rotated_rectangle.exterior.coords

                triangle, heading = getShooting_triangle(viewpoint, min_rotated_rectangle)
                GSV_url = gpano.getGSV_url_frm_lonlat(lon, lat, heading)
                logger.info("GSV url: %s", GSV_url)

                # find intersects in the r-tree
                bound = triangle.bounds
                intersects = r_tree.intersection(bound)
                intersects = list(intersects)

                isIntersected = False
                for inter in intersects:
                    if inter == idx:
                        continue
                    building = buildings[inter]['geometry']
                    building = fionaPolygon2shaple(building)
                    building = shapelyReproject(transformer, building)
                    isIntersected = triangle.intersects(building)
                    if isIntersected:
                        logger.info("Occluded by other houses.")

                if isIntersected:
                    # logger.info("Occluded by other houses.")
                    continue

                ret = gpano.shootLonlat(x, y, polygon=min_rotated_rectangle, saved_path=saved_path, prefix=row_id,
                                        width=w,
                                        height=h, fov=90)
                # logger.info("Google Street View: %s", gpano.getGSV_url_frm_lonlat(lon, lat, ))

                # logger.info("intersects: %s", intersects)

            except Exception as e:
                logger.error("Error in building polygons: %s", e, exc_info=True)
                continue

        # for idx, polygon in enumerate(shoot_ply.polygons):
        #     try:
        #         x, y = polygon.centroid
        #     except Exception as e:
        #         logger.error("Error in enumerate polygons: %s", e)
        #         continue


    except Exception as e:
        logger.error("shoot_houston_building: %s", e)

def shoot_Boston_buildings():
    try:
        shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\boston\building_flood.shp'
        saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\boston\building_squre'

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "info.log"))
        w = 768
        h = 768
        # logger.info(os.path.basename(file))

        logger = logging.getLogger('console_only')


        rtree_path = shape_file.replace(".shp", '_rtree.idx')
        r_tree = None

        buildings = fiona.open(shape_file)

        inEPSG  = 'EPSG:2249'
        outEPSG = 'EPSG:4326'

        if os.path.exists(rtree_path):
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))
            logger.info("Loading the Rtree: %s", rtree_path)
        else:
            logger.info("Creating the Rtree: %s", rtree_path)
            create_rtree(shape_file, inEPSG=inEPSG, outEPSG=outEPSG)

            logger.info("Loading the Rtree: %s", rtree_path)
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))


        # logging.basicConfig(stream=sys.stderr, level=logging.INFO)


        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # generate shaple.Polygons.
        shoot_ply = shoot_polygons([])
        start = 0
        for idx in tqdm(range(start, len(buildings))):
            try:
                building = buildings[idx]
                # logger.info("\n\n")
                print('\n')
                logger.info("Processing polyogn #: %d", idx)
                geometry = building['geometry']['coordinates']

                row_id = building['properties']['ID']
                row_id = int(row_id)

                if len(geometry) > 1:
                    logger.info('Polygon # %s have multiple (%d) parts.', idx, len(geometry))
                    geometry = geometry[:1]
                geometry = np.array(geometry).squeeze(0)

                # coords = polygon.exterior.coords.xy  # coords:

                xs, ys = transformer.transform(geometry[:, 0], geometry[0:, 1])

                polygon = Polygon(zip(xs, ys))

                x, y = polygon.centroid.xy  # x is an array, the number is x[0]
                x = x[0]
                y = y[0]

                logger.info("polygon.centroid: %f, %f", x, y)

                panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)

                if panoId == 0:
                    logger.info("Cannot find a street view image at : %s, %s ", x, y)
                    continue

                viewpoint = np.array((lon, lat))
                # triangle = getShooting_triangle(viewpoint, polygon)

                min_rotated_rectangle = polygon.minimum_rotated_rectangle
                # points_list = min_rotated_rectangle.exterior.coords

                triangle, heading = getShooting_triangle(viewpoint, min_rotated_rectangle)
                GSV_url = gpano.getGSV_url_frm_lonlat(lon, lat, heading)
                logger.info("GSV url: %s", GSV_url)

                # find intersects in the r-tree
                bound = triangle.bounds
                intersects = r_tree.intersection(bound)
                intersects = list(intersects)

                isIntersected = False
                for inter in intersects:
                    if inter == idx:
                        continue
                    building = buildings[inter]['geometry']
                    building = fionaPolygon2shaple(building)
                    building = shapelyReproject(transformer, building)
                    isIntersected = triangle.intersects(building)
                    if isIntersected:
                        logger.info("Occluded by other houses.")
                        break

                if isIntersected:
                    # logger.info("Occluded by other houses.")
                    continue

                ret = gpano.shootLonlat(x, y, polygon=min_rotated_rectangle, saved_path=saved_path, prefix=row_id,
                                        width=w,
                                        height=h, fov=90)
                # logger.info("Google Street View: %s", gpano.getGSV_url_frm_lonlat(lon, lat, ))

                # logger.info("intersects: %s", intersects)

            except Exception as e:
                logger.error("Error in building polygons: %s", e, exc_info=True)
                continue

        # for idx, polygon in enumerate(shoot_ply.polygons):
        #     try:
        #         x, y = polygon.centroid
        #     except Exception as e:
        #         logger.error("Error in enumerate polygons: %s", e)
        #         continue


    except Exception as e:
        logger.error("shoot_houston_building: %s", e)


def shoot_Baltimore_buildings():
    try:
        shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\vacant_house\data\vacant_building.shp'
        saved_path =r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\vacant_house\vacant_images'

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "info.log"))
        w = 768
        h = 1024
        # logger.info(os.path.basename(file))

        logger = logging.getLogger('console_only')


        rtree_path = shape_file.replace(".shp", '_rtree.idx')
        r_tree = None

        buildings = fiona.open(shape_file)

        inEPSG  = 'EPSG:4326'
        outEPSG = 'EPSG:4326'

        if os.path.exists(rtree_path):
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))
            logger.info("Loading the Rtree: %s", rtree_path)
        else:
            logger.info("Creating the Rtree: %s", rtree_path)
            create_rtree(shape_file, inEPSG=inEPSG, outEPSG=outEPSG)

            logger.info("Loading the Rtree: %s", rtree_path)
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))


        # logging.basicConfig(stream=sys.stderr, level=logging.INFO)


        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # generate shaple.Polygons.
        shoot_ply = shoot_polygons([])
        start = 0
        for idx in tqdm(range(start, len(buildings))):
            try:
                building = buildings[idx]
                # logger.info("\n\n")
                print('\n')
                logger.info("Processing polyogn #: %d", idx)
                geometry = building['geometry']['coordinates']

                row_id = building['properties']['row_id_1']
                row_id = int(row_id)
                row_id = str(row_id)

                if len(geometry) > 1:
                    logger.info('Polygon # %s have multiple (%d) parts.', idx, len(geometry))
                    geometry = geometry[:1]
                geometry = np.array(geometry).squeeze(0)

                # coords = polygon.exterior.coords.xy  # coords:

                xs, ys = transformer.transform(geometry[:, 0], geometry[0:, 1])

                polygon = Polygon(zip(xs, ys))

                x, y = polygon.centroid.xy  # x is an array, the number is x[0]
                x = x[0]
                y = y[0]

                logger.info("polygon.centroid: %f, %f", x, y)

                panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)

                if panoId == 0:
                    logger.info("Cannot find a street view image at : %s, %s ", x, y)
                    continue

                viewpoint = np.array((lon, lat))
                # triangle = getShooting_triangle(viewpoint, polygon)

                min_rotated_rectangle = polygon.minimum_rotated_rectangle
                # points_list = min_rotated_rectangle.exterior.coords

                triangle, heading = getShooting_triangle(viewpoint, min_rotated_rectangle)

                triangle = triangle.buffer(-0.000007)  # about 1 m, needs to be improved. !!!!!!!

                GSV_url = gpano.getGSV_url_frm_lonlat(lon, lat, heading)
                logger.info("GSV url: %s", GSV_url)

                # find intersects in the r-tree
                bound = triangle.bounds
                intersects = r_tree.intersection(bound)
                intersects = list(intersects)

                isIntersected = False
                for inter in intersects:
                    if inter == idx:
                        continue
                    building = buildings[inter]['geometry']
                    building = fionaPolygon2shaple(building)
                    building = shapelyReproject(transformer, building)
                    isIntersected = triangle.intersects(building)
                    if isIntersected:
                        logger.info("Occluded by other houses.")
                        break

                if isIntersected:
                    # logger.info("Occluded by other houses.")
                    continue

                ret = gpano.shootLonlat(x, y, polygon=min_rotated_rectangle, saved_path=saved_path, prefix=row_id,
                                        width=w,
                                        height=h, fov=90)
                # logger.info("Google Street View: %s", gpano.getGSV_url_frm_lonlat(lon, lat, ))

                # logger.info("intersects: %s", intersects)

            except Exception as e:
                logger.error("Error in building polygons: %s", e, exc_info=True)
                continue

        # for idx, polygon in enumerate(shoot_ply.polygons):
        #     try:
        #         x, y = polygon.centroid
        #     except Exception as e:
        #         logger.error("Error in enumerate polygons: %s", e)
        #         continue


    except Exception as e:
        logger.error("shoot_houston_building: %s", e)

if __name__ == "__main__":

    # shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\building_in_flood_houson.shp'
    # rtree = rtree

    # shoot_houston_building()
    # shoot_philly_building()

    shoot_oceancity_building()
    # shoot_Boston_buildings()
    # shoot_Boston_buildings()
    # create_rtree(shape_file)
    # shoot_Galveston_buildings()
    # shoot_Baltimore_buildings()