import logging
import numpy as np
import os
import math
import fiona
import random
import yaml
import multiprocessing as mp
from pano import GSV_pano
import random
# import shapefile
import glob
import time
from PIL import Image

from tqdm import tqdm

import datetime
import  utils
import fiona
import json
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
from adjustText import adjust_text
from numpy import linalg as LA

from tqdm import tqdm

import shapely
from shapely.geometry import Point, Polygon

shapely.speedups.disable()


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
logger = logging.getLogger('LOG.file')

logging.shutdown()

def get_panorama(mp_list, saved_path, zoom=4):
    total = len(mp_list)
    saved_path = saved_path
    processed_cnt = 0
    while len(mp_list) > 0:
        try:
            i, lon, lat = mp_list.pop(0)
            pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=saved_path)
            pano1.download_panorama(zoom=zoom)

            processed_cnt = total - len(mp_list)
            print(f"PID {os.getpid()} downloaded row # {i}, {lon}, {lat}, {pano1.panoId}. {processed_cnt} / {total}")

        except Exception as e:
            print(e)
            continue


def downoad_panoramas_from_json_list(json_file_list, saved_path, zoom=4):
    total_cnt = len(json_file_list)
    pre_dir = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_panoramas'
    start_time_all = time.perf_counter()
    while len(json_file_list) > 0:
        try:
            start_time = time.perf_counter()
            json_file = json_file_list.pop()
            basename = os.path.basename(json_file)[:-5] + "_4.jpg"
            new_name = os.path.join(saved_path, basename)
            if os.path.exists(new_name):
                print(f"{basename} exits, continue.")
                continue

            basename = os.path.basename(json_file)[:-5] + "_5.jpg"
            new_name = os.path.join(pre_dir, basename)
            if os.path.exists(new_name):
                print(f"{basename} exits, resample the current file.")
                img_pil = Image.open(new_name)
                w, h = img_pil.size
                w = int(w/2)
                h = int(h/2)
                img_pil = img_pil.resize((w, h))
                zoom4_pano_name = os.path.join(saved_path, os.path.basename(json_file)[:-5] + "_4.jpg")
                img_pil.save(zoom4_pano_name)
                continue
            pano1 = GSV_pano(json_file=json_file, saved_path=saved_path)
            pano1.get_panorama(zoom=zoom)
            total_time = (time.perf_counter() - start_time_all)
            efficency = total_time / (total_cnt - len(json_file_list))
            time_remain = efficency * len(json_file_list)
            print(f"Time spent (seconds): {time.perf_counter() - start_time:.1f}, time used: {utils.delta_time(total_time)} , time remain: {utils.delta_time(time_remain)}  \n")

        except Exception as e:
            logging.error("Error in downoad_panoramas_from_json_list(): %s, %s" % (e, json_file), exc_info=True)
            continue

def download_panos_DC_from_jsons():
    logger.info("Started...")
    saved_path = r'H:\Research\sidewalk_wheelchairs\DC_panoramas_4'
    json_files_path = r'D:\Research\sidewalk_wheelchair\jsons'

    json_files = glob.glob(os.path.join(json_files_path, "*.json"))
    zoom = 4


    # panoIds = [os.path.basename(f)[:-5] for f in json_files]

    # downoad_panoramas_from_json_list(json_files, saved_path, zoom)

    logger.info("Making mp_list...")
    panoIds_mp = mp.Manager().list()

    skips = 24400
    for f in json_files[skips:]:
        panoIds_mp.append(f)


    process_cnt = 10

    pool = mp.Pool(processes=process_cnt)

    for i in range(process_cnt):
        pool.apply_async(downoad_panoramas_from_json_list, args=(panoIds_mp, saved_path, zoom))
    pool.close()
    pool.join()


def download_panos_DC():
    logger.info("Started...")
    saved_path = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_panoramas'
    shp_path = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\road_pts_30m2.shp'
    points = fiona.open(shp_path)

    skips = 9710
    points = points[:]

    logger.info("Making mp_list...")
    lonlats_mp = mp.Manager().list()

    for i in range(len(points) - skips):
        i += skips
        # geometry = points[i]['geometry']['coordinates'] # using fiona
        geometry = points[i].shape.__geo_interface__['coordinates'] # using pyshp

        lon, lat = geometry
        lonlats_mp.append((i, lon, lat))
    logger.info("Finished mp_list (%d records).", len(lonlats_mp))

    cut_point = 100000
    lonlats_mp_first100 = lonlats_mp[:cut_point]
    random.shuffle(lonlats_mp_first100)
    lonlats_mp[:cut_point] = lonlats_mp_first100

    random.shuffle(lonlats_mp)

    process_cnt = 5
    pool = mp.Pool(processes=process_cnt)

    for i in range(process_cnt):
        pool.apply_async(get_panorama, args=(lonlats_mp, saved_path))
    pool.close()
    pool.join()

    # for i in tqdm(range(len(points))):
    #     geometry = points[i]['geometry']['coordinates']
    #     lon, lat = geometry
    #     logger.info("Processing row #: %d, %f, %f", i, lon, lat)
    #     get_panorama(lon, lat, saved_path)




def get_DOMs():
    # lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
    # full_path = r'J:\Research\StreetView\gsv_pano\AZK1jDGIZC1zmuooSZCzEg.png'
    # full_path = r'D:\Code\StreetView\gsv_pano\-0D29S37SnmRq9Dju9hkqQ.png'
    # panoId_2019 = "-0D29S37SnmRq9Dju9hkqQ"
    seg_dir = r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_segmented'
    seg_files = glob.glob(os.path.join(seg_dir, "*.png"))

    saved_path = r"K:\OneDrive_USC\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_DOMs"

    resolution = 0.05
    total_cnt = len(seg_files)

    seg_files.reverse()
    random.shuffle(seg_files)

    while len(seg_files) > 0:
        seg_file = seg_files.pop()
    # for idx, seg_file in enumerate(seg_files[1:]):
        start_time = time.perf_counter()
        try:
            print("Processing: ", total_cnt - len(seg_files), seg_file)
            panoId = os.path.basename(seg_file)[:-4]

            pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=saved_path)

            new_name = os.path.join(saved_path, panoId + f"_DOM_{resolution:.2f}.tif")
            is_processed = os.path.exists(new_name)

            Links = pano1.jdata['Links']
            for link in Links:
                temp_name = os.path.join(seg_dir, link['panoId'] +'.png')
                if temp_name in seg_files:
                    seg_files.remove(temp_name)
                    seg_files.append(temp_name)
                #
                # else:
                #     if (not os.path.exists(temp_name)):
                #         pass

            if is_processed:
                print("Skip: ", seg_file)
                continue

            # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=r'J:\Research\StreetView\gsv_pano\test_results')
            pano1.set_segmentation_path(full_path=seg_file)
            DOM = pano1.get_DOM(width=40, height=40, resolution=resolution, zoom=4, img_type='segmentation',  fill_clipped_seg=True)

            print("Time spent (seconds): ", time.perf_counter() - start_time, '\n')
            # palette = Image.open(seg_file).getpalette()
            # palette = np.array(palette,dtype=np.uint8)

            # pil_img = PIL.Image.fromarray(DOM['DOM'])
            # pil_img.putpalette(palette)
            # self.assertEqual((800, 800, 3), DOM.shape)
            # pil_img.show()
        except Exception as e:
            print("Error :", e, seg_file)
            continue


def get_DOMs():

    seg_dir = r'D:\Research\sidewalk_wheelchair\DC_segmented'
    seg_files = glob.glob(os.path.join(seg_dir, "*.png"))
    random.shuffle(seg_files)

    saved_path = r"D:\Research\sidewalk_wheelchair\DC_DOMs"

    resolution = 0.05

    seg_files_mp = mp.Manager().list()
    for f in seg_files:
        seg_files_mp.append(f)

    process_cnt = 18
    pool = mp.Pool(processes=process_cnt)

    for i in range(process_cnt):
        pid_id = i
        pool.apply_async(get_DOM, args=(pid_id, seg_files_mp, saved_path, resolution))
    pool.close()
    pool.join()

def quick_DOM():

    seg_dir = r'D:\DC_segmented'
    pano_dir = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_panoramas'
    seg_files = glob.glob(os.path.join(seg_dir, "*.png"))
    pano_files = glob.glob(os.path.join(pano_dir, "*.jpg"))
    saved_path = r'F:\Research\sidewalk_wheelchair\DOMs'
    resolution = 0.05

    img_w = 40
    img_h = 40
    zoom = 4

    # pano_files.reverse()
    # pano_files = pano_files[:]

    for idx, pano_file in enumerate(pano_files[120000:]):

        try:

            print(f"Processing: {idx} / {len(pano_files)}, {pano_file}")
            panoId = os.path.basename(pano_file)[:-6]

            new_name = os.path.join(saved_path, f"{panoId}_DOM_{resolution:0.2f}.tif")
            if os.path.exists(new_name):
                print(f"Skip processed panoramas: {panoId}")
                continue

            img_zoom = int(pano_file[-5])

            if img_zoom == 4:
                print("Skipe: ", pano_file)
                continue

            # if img_zoom == 5:
            #     img_zoom5 = Image.open(pano_file)
            #     pano_file = pano_file.replace("_5.jpg", "_4.jpg")

            timer_start = time.perf_counter()

            distance_threshole = img_w * 1.5


            pano1 = GSV_pano(panoId=panoId, saved_path=pano_dir, crs_local=6487)

            pano1.get_depthmap(zoom=zoom)

            pano1.depthmap['ground_mask'] = np.where(pano1.depthmap['depthMap'] < distance_threshole, 1, 0)
            mask_h, mask_w = pano1.depthmap['ground_mask'].shape
            pano1.depthmap['ground_mask'][int(mask_h / 4 * 3):, :] = 0



            P = pano1.get_ground_points(zoom=zoom, color=True, img_type="pano")



            # P = P[P[:, 3] < distance_threshole]
            P = P[P[:, 0] < img_w/2]
            P = P[P[:, 0] > -img_w/2]
            P = P[P[:, 1] < img_h/2]
            P = P[P[:, 1] > -img_h/2]

            timer_end = time.perf_counter()


            np_img, worldfile = pano1.points_to_DOM(P[:, 0], P[:, 1], P[:, 4:7], resolution=resolution)

            pano1.save_image(np_img, new_name, worldfile)

            print("Time spent (second):", timer_end - timer_start)
        except Exception as e:
            print("Error in quick_DOM():", pano_file, e)
        #
        # v = pptk.viewer(P[:, :3])
        # v.set(point_size=0.001, show_axis=True, show_grid=False)
        # v.attributes(P[:, 4:7]/255.0)

if __name__ == '__main__':
    download_panos_DC()
    # get_DOMs()
    # quick_DOM()