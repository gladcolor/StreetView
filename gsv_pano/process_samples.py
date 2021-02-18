import logging
import os
import math
import fiona
import yaml
import multiprocessing as mp
from pano import GSV_pano
import random

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
    saved_path = saved_path
    while len(mp_list) > 0:
        try:
            i, lon, lat = mp_list.pop(0)
            pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=saved_path)
            pano1.download_panorama(zoom=5)

            print(f"PID {os.getpid()} downloaded row # {i}, {lon}, {lat}, {pano1.panoId}")
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
        geometry = points[i]['geometry']['coordinates']
        lon, lat = geometry
        lonlats_mp.append((i, lon, lat))
    logger.info("Finished mp_list (%d records).", len(lonlats_mp))

    cut_point = 100000
    lonlats_mp_first100 = lonlats_mp[:cut_point]
    random.shuffle(lonlats_mp_first100)
    lonlats_mp[:cut_point] = lonlats_mp_first100

    process_cnt = 10
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







if __name__ == '__main__':
    download_panos_DC()