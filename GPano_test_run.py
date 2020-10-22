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
import Shoot_objects

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import shapely.wkt
import os
import fiona
from pyproj import Proj, transform, Transformer

from tqdm import tqdm

import logging
import sys
import pprint

from rtree import index

from pyproj import Proj, transform
from geopy.distance import geodesic


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

def download_buildings_boston():
    shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\boston\building_flood.shp'

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    buildings = fiona.open(shape_file)

    saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\boston\building_images3'

    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    # os.system("mkdir I:\\Research\\House\\images\\1.0 I:\\Research\\House\\images\\1.5 I:\\Research\\House\\images\\2.0 I:\\Research\\House\\images\\2.5 I:\\Research\\House\\images\\3.0 I:\\Research\\House\\images\\3.5 I:\\Research\\House\\images\\4.0 I:\\Research\\House\\images\\4.5")

    # Process_cnt = 10
    # pool = mp.Pool(processes=Process_cnt)https://m.tsemporium.com/zh_cn/
    #
    # for i in range(Process_cnt):
    #     pool.apply_async(self.getPanoJPGfrmArea, args=(yaw_list, seed_pts_mp, saved_path, boundary_vert, zoom))
    # pool.close()
    # pool.join()

    center_x = 0;
    centre_y = 0;


    for i in tqdm(range(len(buildings))):
        # FID, area_m, ACCTID, story, GEOID, tract_pop,lon, lat, geometry = buildings.iloc[i]
        # geometry = shapely.wkt.loads(geometry)


        try:
            geometry = buildings[i]['geometry']['coordinates']
            print("Processing (FID): ", i)
            geometry = np.array(geometry).squeeze(0)

            x = geometry[:, 0].mean()
            y = geometry[:, 1].mean()

            inProj = Proj('epsg:2249')  # NAD83 / Massachusetts Mainland (ftUS)
            outProj = Proj('epsg:4326')

            # lon, lat = transform(inProj, outProj, x, y)

            geometry = transform(inProj, outProj, geometry[:, 0], geometry[:, 1])  # return lat, lon

            geometry = Polygon(zip(geometry[1], geometry[0]))

            lat, lon = geometry.centroid.y, geometry.centroid.x

            distance_threshold = 50 # meter

            ret = gpano.shootLonlat(lon, lat, polygon=geometry, saved_path=saved_path, prefix='', width=576, height=768,  fov=90, distance_threshold=distance_threshold)

             # shootLonlat(self, ori_lon, ori_lat, saved_path='', views=1, prefix='', suffix='', width=1024,
             #                height=768, pitch=0):
        except Exception as e:
            print("Error in download_buildings_boston(): ", e, i)

def download_buildings_houston():
    shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\building_in_flood_houson.shp'

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    buildings = fiona.open(shape_file)

    saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\houston\street_images'

    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    # os.system("mkdir I:\\Research\\House\\images\\1.0 I:\\Research\\House\\images\\1.5 I:\\Research\\House\\images\\2.0 I:\\Research\\House\\images\\2.5 I:\\Research\\House\\images\\3.0 I:\\Research\\House\\images\\3.5 I:\\Research\\House\\images\\4.0 I:\\Research\\House\\images\\4.5")

    # Process_cnt = 10
    # pool = mp.Pool(processes=Process_cnt)https://m.tsemporium.com/zh_cn/
    #
    # for i in range(Process_cnt):
    #     pool.apply_async(self.getPanoJPGfrmArea, args=(yaw_list, seed_pts_mp, saved_path, boundary_vert, zoom))
    # pool.close()
    # pool.join()

    center_x = 0;
    centre_y = 0;


    for i in tqdm(range(len(buildings))):
        # FID, area_m, ACCTID, story, GEOID, tract_pop,lon, lat, geometry = buildings.iloc[i]
        # geometry = shapely.wkt.loads(geometry)


        try:
            geometry = buildings[i]['geometry']['coordinates']
            print("Processing (FID): ", i)
            geometry = np.array(geometry).squeeze(0)

            x = geometry[:, 0].mean()
            y = geometry[:, 1].mean()

            inProj = Proj('epsg:3857')  # NAD83 / Massachusetts Mainland (ftUS)
            outProj = Proj('epsg:4326')

            # lon, lat = transform(inProj, outProj, x, y)

            geometry = transform(inProj, outProj, geometry[:, 0], geometry[:, 1])  # return lat, lon

            geometry = Polygon(zip(geometry[1], geometry[0]))

            lat, lon = geometry.centroid.y, geometry.centroid.x

            distance_threshold = 50 # meter

            ret = gpano.shootLonlat(lon, lat, polygon=geometry, saved_path=saved_path, prefix='', width=576, height=768,  fov=90, distance_threshold=distance_threshold)

             # shootLonlat(self, ori_lon, ori_lat, saved_path='', views=1, prefix='', suffix='', width=1024,
             #                height=768, pitch=0):
        except Exception as e:
            print("Error in download_buildings_boston(): ", e, i)


def shoot_baltimore():
    csv_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\vacant_house\data\Vacant_Buildings.csv'
    saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\vacant_house\street_images'

    df = pd.read_csv(csv_file)
    print("All rows: ", len(df))

    # lonlats_mp = mp.Manager().list()
    for idx, row in df[:].iterrows():
        prefix = idx + 1
        lat, lon = row['Location'].split(',')
        # lonlats_mp.append((lon, lat, idx + 1))
        print("Processing: ", idx)
        gpano.shootLonlat(lon, lat, saved_path=saved_path, prefix=prefix, views=1)

    # gpano.shootLonlats_mp(lonlats_mp, saved_path, Process_cnt=1, views=3, suffix='', width=1024, height=768)

def download_buildings_baltimore():
    shape_file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\vacant_house\data\non_vacant_buildings.shp'

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    buildings = fiona.open(shape_file)

    saved_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\vacant_house\non_vacant_images'

    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    # os.system("mkdir I:\\Research\\House\\images\\1.0 I:\\Research\\House\\images\\1.5 I:\\Research\\House\\images\\2.0 I:\\Research\\House\\images\\2.5 I:\\Research\\House\\images\\3.0 I:\\Research\\House\\images\\3.5 I:\\Research\\House\\images\\4.0 I:\\Research\\House\\images\\4.5")

    # Process_cnt = 10
    # pool = mp.Pool(processes=Process_cnt)https://m.tsemporium.com/zh_cn/
    #
    # for i in range(Process_cnt):
    #     pool.apply_async(self.getPanoJPGfrmArea, args=(yaw_list, seed_pts_mp, saved_path, boundary_vert, zoom))
    # pool.close()
    # pool.join()

    center_x = 0;
    centre_y = 0;

    buildings = buildings[:]
    for i in tqdm(range(len(buildings))):
        # FID, area_m, ACCTID, story, GEOID, tract_pop,lon, lat, geometry = buildings.iloc[i]
        # geometry = shapely.wkt.loads(geometry)


        try:
            geometry = buildings[i]['geometry']['coordinates']
            row_id = buildings[i]['properties']['objectid_1']
            row_id = int(row_id)
            row_id = str(row_id)
            print("Processing (FID): ", i)
            geometry = np.array(geometry).squeeze(0)

            x = geometry[:, 0].mean()
            y = geometry[:, 1].mean()

            # inProj = Proj('epsg:2249')  # NAD83 / Massachusetts Mainland (ftUS)
            # outProj = Proj('epsg:4326')

            # lon, lat = transform(inProj, outProj, x, y)

            # geometry = transform(inProj, outProj, geometry[:, 0], geometry[:, 1])  # return lat, lon

            geometry = Polygon(geometry)

            lat, lon = geometry.centroid.y, geometry.centroid.x

            distance_threshold = 100

            ret = gpano.shootLonlat(lon, lat, polygon=geometry, saved_path=saved_path, prefix=str(row_id), width=768, height=768,  fov=90, distance_threshold=distance_threshold)
            logging.info("row_id: %s", row_id)
             # shootLonlat(self, ori_lon, ori_lat, saved_path='', views=1, prefix='', suffix='', width=1024,
             #                height=768, pitch=0):
        except Exception as e:
            print("Error in download_buildings_baltimore(): ", e, i)

def clip_panos_ocean_city():
    folder = r'J:\Research\Resilience\data\OcenaCity\panos2'
    saved_path = r'J:\Research\Resilience\data\OcenaCity\pano_clips'
    fov = 90   # degree
    h_w_ratio = 3/4
    yawList = [90, 270]
    files = glob.glob(os.path.join(folder, "*.jpg"))
    for idx, jpg in tqdm(enumerate(files)):
        basename = os.path.basename(jpg)
        # img = Image.open(jpg)
        # img = np.array(img)
        img = cv2.imread(jpg)
        h_img, w_img, channel = img.shape
        w = int(fov/360 * w_img)
        h = int(w * h_w_ratio)
        print("panorama shape:", img.shape)
        fov_h = radians(90)

        theta0 = 0
        pano_pitch = 0

        fov_v = atan((h * tan((fov_h / 2)) / w)) * 2
        for yaw in yawList:
            rimg = gpano.clip_pano(theta0,
                                   radians(yaw),
                                   fov_h,
                                   fov_v,
                                   w,
                                   img,
                                   pano_pitch)
            new_name = os.path.join(saved_path, basename)
            cv2.imwrite('%s_%d_%d.jpg' % (new_name, theta0, yaw), rimg)


def get_panoIDs_ocean_city_address():
    address_txt = r'X:\My Drive\Research\StreetGraph\data\oceancity\property_address.txt'
    folder = os.path.dirname(address_txt)
    new_name = address_txt[:-4] + "_panoIds.csv"
    saved_csv = os.path.join(folder, new_name)

    f = open(address_txt, 'r')
    addresses = f.readlines()
    f.close()

    f = open(saved_csv, 'w')
    f.writelines("addr,panoIs,lon,lat" + '\n')

    for idx, addr in tqdm(enumerate(addresses)):
        addr = addr.replace("\n", '')
        try:
             panoIds, lon, lat = gpano.getPanoIDfrmAddress(addr)
             print(f"{addr} : {panoIds}, {lon}, {lat} \n")
             f.writelines(','.join([addr,str(panoIds), str(lon), str(lat)]) + "\n")
        except Exception as e:
            print("Error in get_panoIDs_ocean_city_address(), address, log:", addr, e)
            continue

    f.close()

def get_PanoJPG_oceancity():
    task_txt = r'X:\My Drive\Research\StreetGraph\data\oceancity\missing_panos.txt'
    folder = os.path.dirname(task_txt)

    saved_path = r'X:\My Drive\Research\StreetGraph\data\oceancity\panos'

    f = open(task_txt, 'r')
    tasks = f.readlines()
    f.close()


    for idx, task in tqdm(enumerate(tasks)):
        panoId = task.replace("\n", '')
        try:
            gpano.getPanoJPGfrmPanoId(panoId, saved_path, zoom=5)
            print(f"Processing: {idx}, {panoId}   \n")

        except Exception as e:
            print("Error in get_PanoJPG_oceancity, address, log:", panoId, e)
            continue

    f.close()



def get_PanoJPG_Hampton():
    shape_file = r'K:\Dataset\Elevation_certificates\Hampton_Roads_Elevation_Certificates__NAVD_88_-shp\Hampton_Roads_Elevation_Certificates__NAVD_88_.shp'
    saved_path = r'K:\Dataset\HamptonRoads\panoramas'

    buildings = fiona.open(shape_file)


    for idx, building in tqdm(enumerate(buildings)):

        geometry = building['geometry']['coordinates']

        ID = str(building['properties']['ID'])

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "_info.log"))

        inEPSG = 'EPSG:2925'
        outEPSG = 'EPSG:4326'

        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)
        geometry = np.array(geometry).squeeze(0)

        xs, ys = transformer.transform(geometry[:, 0], geometry[0:, 1])

        polygon = Polygon(zip(xs, ys))

        x, y = polygon.centroid.xy  # x is an array, the number is x[0]
        x = x[0]
        y = y[0]

        logger.info("polygon.centroid: %f, %f", x, y)

        panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)


        try:
            gpano.getJsonfrmPanoID(panoId, dm=1, saved_path=saved_path)
            gpano.getPanoJPGfrmPanoId(panoId, saved_path=saved_path, zoom=5)

            print(f"Processing: {idx}, {panoId}   \n")

        except Exception as e:
            print("Error in get_PanoJPG_Hampton, panoId, log:", panoId, e)
            continue




def clip_panos_oceancity_with_panoId():
    try:
        shape_file = r'X:\My Drive\Research\StreetGraph\data\oceancity\vectors\ocean_parcel_panoid.shp'
        saved_path = r'X:\My Drive\Research\StreetGraph\data\oceancity\images3'

        pano_path = r'X:\My Drive\Research\StreetGraph\data\oceancity\panos'

        setup_logging(yaml_path, logName=shape_file.replace(".shp", "_info.log"))

        inEPSG = 'EPSG:3424'
        outEPSG = 'EPSG:4326'

        fov_h_degree = 90  # degree
        fov_h_radian = radians(fov_h_degree)

        h_w_ratio = 1

        rtree_path = shape_file.replace(".shp", '_rtree.idx')
        r_tree = None

        if os.path.exists(rtree_path):
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))
            logger.info("Loading the Rtree: %s", rtree_path)
        else:
            logger.info("Creating the Rtree: %s", rtree_path)
            Shoot_objects.create_rtree(shape_file, inEPSG=inEPSG, outEPSG=outEPSG)

            logger.info("Loading the Rtree: %s", rtree_path)
            r_tree = index.Rtree(rtree_path.replace(".idx", ''))

        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        buildings = fiona.open(shape_file)

        transformer = Transformer.from_crs(inEPSG, outEPSG, always_xy=True)

        # generate shaple.Polygons.

        start = 10000
        for idx in tqdm(range(start, len(buildings))):
            try:
                building = buildings[idx]

                geometry = building['geometry']['coordinates']
                ID = str(building['properties']['ID'])

                logger.info("Processing polyogn # ID: %s", str(ID))

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

                # panoId, lon, lat = gpano.getPanoIDfrmLonlat(x, y)

                panoId = building['properties']['panoIds']
                lon = building['properties']['lon']
                lat = building['properties']['lat']

                if panoId == 0:
                    logger.info("Cannot find a street view image at : %s, %s ", x, y)
                    continue

                viewpoint = np.array((lon, lat))
                # triangle = getShooting_triangle(viewpoint, polygon)

                min_rotated_rectangle = polygon.minimum_rotated_rectangle
                # points_list = min_rotated_rectangle.exterior.coords



                # triangle, heading = Shoot_objects.getShooting_triangle(viewpoint, min_rotated_rectangle)
                heading = gpano.getDegreeOfTwoLonlat(lat, lon, y, x)
                heading = round(heading, 2)
                GSV_url = gpano.getGSV_url_frm_lonlat(lon, lat, heading)
                logger.info("GSV url: %s", GSV_url)

                # find intersects in the r-tree
                # bound = triangle.bounds
                # intersects = r_tree.intersection(bound)
                # intersects = list(intersects)

                # isIntersected = False
                # for inter in intersects:
                #     if inter == idx:
                #         continue
                #     building = buildings[inter]['geometry']
                #     building = Shoot_objects.fionaPolygon2shaple(building)
                #     building = Shoot_objects.shapelyReproject(transformer, building)
                #     isIntersected = triangle.intersects(building)
                #     if isIntersected:
                #         logger.info("Occluded by other houses.")
                #         break
                #
                # if isIntersected:
                #     logger.info("Occluded by other houses.")
                #     continue

                json_file = os.path.join(pano_path, panoId + ".json")
                jdata = json.load(open(json_file, 'r'))
                pano_yaw = jdata["Projection"]['pano_yaw_deg']
                pano_yaw = float(pano_yaw)
                phi = float(heading) - pano_yaw

                if phi > math.pi:
                    phi = phi - 2 * math.pi
                if phi < -math.pi:
                    phi = phi + 2 * math.pi
                phi = round(phi, 2)

                car_heading = pano_yaw

                _, fov = gpano.get_fov4edge((lon, lat), car_heading, polygon, saved_path=saved_path,
                                                  file_name=str(ID) + "_" + panoId + "_" + str(
                                                      heading) + '_shooting.png')

                fov_h_degree = fov  # degree
                fov_h_radian = radians(fov_h_degree)


                # open panorama image
                pano_file = os.path.join(pano_path, panoId + ".jpg")
                img = cv2.imread(pano_file)
                h_img, w_img, channel = img.shape
                w = int(fov_h_degree / 360 * w_img)
                h = int(w * h_w_ratio)
                fov_v_radian = atan((h * tan((fov_h_radian / 2)) / w)) * 2
                print("panorama shape:", img.shape)




                theta0 = 0
                pano_pitch = 0

                rimg = gpano.clip_pano(theta0,
                                       radians(phi),
                                       fov_h_radian,
                                       fov_v_radian,
                                       w,
                                       img,
                                       pano_pitch)
                basename = f"{ID}_{panoId}_{str(heading)}.jpg"
                new_name = os.path.join(saved_path, basename)
                cv2.imwrite(new_name, rimg)
                # logger.info("Google Street View: %s", gpano.getGSV_url_frm_lonlat(lon, lat, ))

                logger.info("Clipped: %s", new_name)

            except Exception as e:
                logger.error("Error in building polygons: %s", e)
                continue




    except Exception as e:
        logger.error("shoot_philly_building: %s", e)

def save_depthmap_frm_panoId(panoId, saved_path):
    gsv.saveDepthMap_frm_panoId(panoId, saved_path)

def convert_row_col_to_shperical(theta0, phi0, fov_h, height, width, tilt_pitch=None, tilt_yaw=None):
    return gsv.castesian_to_shperical0(theta0, phi0, tilt_pitch, tilt_yaw, fov_h, height, width)
    # castesian_to_shperical0(self, theta0, phi0, tilt_pitch, tilt_yaw, fov_h, height, width):  # yaw: set the heading, pitch


if __name__ == '__main__':

    # result = convert_row_col_to_shperical(0, math.radians(90), math.radians(90), 4096, 4096)
    # print(result.shape)
    #
    # beta = result[2101, 1491, 0]
    # alpha = result[1856, 1491, 0]
    # print(beta)
    # print(alpha)
    #
    # print("beta:", math.degrees(beta))
    # print("alpha:", math.degrees(alpha))

    get_PanoJPG_Hampton()

    print("Done")

    # save_depthmap_frm_panoId("yrCaFogoA9xZBBydIBVaKQ", r"/media/huan/SSD/Datasets/ocean_city/temp")

    # clip_panos_oceancity_with_panoId()



    # clip_panos_ocean_city()
    # test_getPanoJPGfrmArea()
    # download_buildings()
    # test_getPanoJPGfrmArea_philly()
    # download_buildings_boston()
    # download_buildings_houston()
    # shoot_baltimore()
    # download_buildings_baltimore()

