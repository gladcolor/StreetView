import logging
import os
import math
import fiona
import yaml
import multiprocessing as mp
from pano import GSV_pano
import random
import shapefile
import glob

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
    seg_dir = r'D:\DC_segmented'
    seg_files = glob.glob(os.path.join(seg_dir, "*.png"))

    for idx, seg_file in enumerate(seg_files[0:]):
        panoId = os.path.basename(seg_file)[:-4]
        pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=r"D:\Code\StreetView\gsv_pano\test_results")
        # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=r'J:\Research\StreetView\gsv_pano\test_results')
        pano1.set_segmentation_path(full_path=seg_file)
        DOM = pano1.get_DOM(width=40, height=40, resolution=0.05, zoom=4, img_type='segmentation',  fill_clipped_seg=True)
        # palette = Image.open(seg_file).getpalette()
        # palette = np.array(palette,dtype=np.uint8)

        # pil_img = PIL.Image.fromarray(DOM['DOM'])
        # pil_img.putpalette(palette)
        # self.assertEqual((800, 800, 3), DOM.shape)
        # pil_img.show()

if __name__ == '__main__':
    # download_panos_DC()
    get_DOMs()