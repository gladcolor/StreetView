import logging
import numpy as np
import os
import math
import fiona
import yaml
import multiprocessing as mp
from pano import GSV_pano
import random
# import shapefile
import glob
import time
from PIL import Image

from tqdm import tqdm




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


def get_panorama(mp_list, saved_path):
    total = len(mp_list)
    saved_path = saved_path
    processed_cnt = 0
    while len(mp_list) > 0:
        try:
            i, lon, lat = mp_list.pop(0)
            pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=saved_path)
            pano1.download_panorama(zoom=5)

            processed_cnt = total - len(mp_list)
            print(f"PID {os.getpid()} downloaded row # {i}, {lon}, {lat}, {pano1.panoId}. {processed_cnt} / {total}")

        except Exception as e:
            print(e)
            continue

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

    for idx, seg_file in enumerate(seg_files[1:]):
        try:
            print("Processing: ", idx, seg_file)
            panoId = os.path.basename(seg_file)[:-4]
            pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=r"K:\OneDrive_USC\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_DOMs")
            # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=r'J:\Research\StreetView\gsv_pano\test_results')
            pano1.set_segmentation_path(full_path=seg_file)
            DOM = pano1.get_DOM(width=40, height=40, resolution=0.05, zoom=4, img_type='segmentation',  fill_clipped_seg=True)
            # palette = Image.open(seg_file).getpalette()
            # palette = np.array(palette,dtype=np.uint8)

            # pil_img = PIL.Image.fromarray(DOM['DOM'])
            # pil_img.putpalette(palette)
            # self.assertEqual((800, 800, 3), DOM.shape)
            # pil_img.show()
        except Exception as e:
            print("Error :", e, seg_file)
            continue

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
    # download_panos_DC()
    # get_DOMs()
    quick_DOM()