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
import datetime
import  utils
import fiona
import json
import geopandas as gpd
from PIL import Image

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

def get_DOM(pid_id, seg_files, saved_path, resolution):
    seg_dir = os.path.dirname(seg_files[0])
    total_cnt = len(seg_files)
    start_time_all = time.perf_counter()
    while len(seg_files) > 0:
        seg_file = seg_files.pop()
        start_time = time.perf_counter()
        try:
            print("Process No.", pid_id, "is processing: ", total_cnt - len(seg_files), seg_file)
            panoId = os.path.basename(seg_file)[:-4]

           # pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=saved_path)

            new_name = os.path.join(saved_path, panoId + f"_DOM_{resolution:.2f}.tif")
            is_processed = os.path.exists(new_name)

            Links = pano1.jdata['Links']
            for link in Links:
                temp_name = os.path.join(seg_dir, link['panoId'] +'.png')
                if temp_name in seg_files:
                   seg_files.remove(temp_name)
                   seg_files.append(temp_name)

            if is_processed:
                continue
            pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=saved_path)
            # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=r'J:\Research\StreetView\gsv_pano\test_results')
            pano1.set_segmentation_path(full_path=seg_file)
            DOM = pano1.get_DOM(width=40, height=40, resolution=resolution, zoom=4, img_type='segmentation',  fill_clipped_seg=True)
            total_time = (time.perf_counter() - start_time_all)
            def delta_time(seconds):
                delta1 = datetime.timedelta(seconds=seconds)
                str_delta1 = str(delta1)
                decimal_digi = 0
                point_pos = str_delta1.rfind(".")
                str_delta1 = str_delta1[:point_pos]
                return str_delta1
            efficency = total_time / (total_cnt - len(seg_files))
            time_remain = efficency * len(seg_files)
            print(f"Time spent (seconds): {time.perf_counter() - start_time:.1f}, time used: {delta_time(total_time)} , time remain: {delta_time(time_remain)}  \n")
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

    for idx, pano_file in enumerate(pano_files[20000:]):

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

def merge_measurements():
    sorted_file = r'D:\Research\sidewalk_wheelchair\sorted_panoIds.txt'
    saved_file = r'D:\Research\sidewalk_wheelchair\widths_all2.txt'
    widths_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens'
    sorted_panoIds = open(sorted_file, 'r').readlines()
    sorted_panoIds = [x[:-1] for x in sorted_panoIds][:]



    with open(saved_file, 'w') as f:
        f.writelines('center_x,center_y,length,col,row,end_x,end_y\n')
        for idx, panoId in enumerate(sorted_panoIds):
            try:
                file_name = os.path.join(widths_dir, f'{panoId}_widths.txt')
                if not os.path.exists(file_name):
                    print("No width measurement!")
                    continue
                print(f'Processed {idx} / {len(sorted_panoIds)}')
                lines = open(file_name, 'r').readlines()[1:]
                f.writelines(''.join(lines))
            except Exception as e:
                print("Error in merge_measurements:", e)
                continue


    # print(sorted_panoIds)

def down_panos_in_area(polyon, saved_path='', col_cnt=100, row_cnt=100, json=True, pano=False, pano_zoom=0, depthmap=False, process_cnt=3):
    '''
    Download street view images according to the given polygon
    :param polyon:
    :param json:
    :param pano:
    :param pano_zoom:
    :param depthmap:
    :param process_cnt:
    :return:
    '''
    min_x, min_y, max_x, max_y = polyon.bounds
    # col_cnt = 100
    # row_cnt = 100

    x_scale = np.linspace(min_x, max_x, col_cnt)
    y_scale = np.linspace(min_y, max_y, row_cnt)
    xv, yv = np.meshgrid(x_scale, y_scale)

    seed_points = gpd.GeoDataFrame(zip(xv.ravel(), yv.ravel()), columns={"lon":float, "lat":float}, geometry=gpd.points_from_xy(xv.ravel(), yv.ravel()))

    in_points = seed_points[seed_points.within(polyon)]

    seed_points_mp = mp.Manager().list()
    for idx, row in in_points.iterrows():
        seed_points_mp.append((row['lon'], row['lat']))

    print("Start...")
    print(f"Generate {len(in_points)} in polygon.bounds {polyon.bounds}")

    # random.shuffle(seed_points_mp)

    pending_panoId_mp = mp.Manager().list()

    pending_panoId = collect_links_from_panoramas_mp(saved_path)

    for p in pending_panoId:
        pending_panoId_mp.append(p)

    pool = mp.Pool(processes=process_cnt)

    print(f'Started to download panoramas using {len(seed_points)} seed points.')
    print(f'Processes used: {process_cnt}.')

    for i in range(process_cnt):
        pool.apply_async(download_panoramas_from_seed_points, args=(seed_points_mp, pending_panoId_mp, saved_path, polyon, math.inf, True))
    pool.close()
    pool.join()

    # while len(seed_points_mp) > 0:
        # seed_point = seed_points_mp.pop()
        # download_panoramas_from_seed_points(seed_point=seed_points_mp, saved_path=saved_path, polygon=polyon)

    return in_points

def download_panoramas_from_seed_points(seed_points, pending_panoIds, saved_path, polygon, max_step=math.inf, json_file=True, pano_jpg=False, pano_zoom=0, depthmap=False):
    total_cnt = len(seed_points)
    while len(seed_points) > 1:

        print(f"Processed {len(seed_points)} / {total_cnt} seed points.")

        seed_point = seed_points.pop()

        # downloaded_cnt = 0

        lon = seed_point[0]
        lat = seed_point[1]
        pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=saved_path)

        if pano1.panoId == 0:
            continue

        pending_panoIds.append(pano1.panoId)

        step = 0

        while len(pending_panoIds) > 0:
            try:
                panoId = pending_panoIds.pop()

                json_name = os.path.join(saved_path, panoId + ".json")
                if os.path.exists(json_name):
                    # pano2 = GSV_pano(json_file=json_name)
                    continue
                else:
                    pano2 = GSV_pano(panoId=panoId)
                    print("Downloaded: ", pano2.panoId)


                lon = pano2.lon
                lat = pano2.lat
                pt = Point(lon, lat)
                if pt.within(polygon) or (step < max_step):
                    with open(json_name, 'w') as f:
                        json.dump(pano2.jdata, f)
                        # downloaded_cnt += 1
                    step += 1
                    links = pano2.jdata["Links"]
                    for link in links:
                        link_panoId = link['panoId']
                        if link_panoId in pending_panoIds:
                            pending_panoIds.remove(link_panoId)
                        pending_panoIds.append(link_panoId)

                        # print(link_panoId)
                else:
                    continue



            except Exception as e:
                print("Error in download_panoramas_from_seed_point():", e)
                continue

        print(f"Downloaded {step} panoramas for this seed point.")

def collect_links_from_panoramas_mp(json_dir, process_cnt=6):
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    panoIds = [os.path.basename(f)[:-5] for f in json_files]
    if len(json_files) < 5 * process_cnt:
        process_cnt = 1

    panoIds_mp = mp.Manager().list()
    for panoId in panoIds:
        panoIds_mp.append(panoId)

    pending_panoIds_mp = mp.Manager().list()

    print(f"Started to collect undownloaded adjacent jsons in: {json_dir}")
    print(f"Processes used: {process_cnt}. \nPlease wait...")

    pool = mp.Pool(processes=process_cnt)
    for i in range(process_cnt):
        pool.apply_async(find_neighors_from_json_files, args=(panoIds_mp, pending_panoIds_mp, json_dir))
    pool.close()
    pool.join()

    print(f"Finished collecing, obtain {len(pending_panoIds_mp)} adjacent jsons from {len(json_files)} json files.")

    return pending_panoIds_mp

    # panoIds
    # pass

def find_neighors_from_json_files(panoIds, pending_Ids, json_dir):
    # print(len(panoIds))
    panoIds_static = list(panoIds)
    # print(len(panoIds))
    # print(len(panoIds_static))

    while len(panoIds) > 1:
        panoId = panoIds.pop()

        # print(f"Processed {len(panoIds_static) - len(panoIds)} / {len(panoIds_static)} seed points.")
    # for panoId in panoIds:
        # print(panoId)
        new_name = os.path.join(json_dir, panoId + ".json")
        try:
            jdata = json.load(open(new_name, 'r'))
            Links = jdata['Links']
            for link in Links:
                # print(link)
                link_panoId = link['panoId']
                # print(link_panoId)
                # if link_panoId in sorted_panoIds:
                #     panoIds.append(link_panoId)
                if not link_panoId in panoIds_static:
                    pending_Ids.append(link_panoId)
                    # print(link_panoId)


        except Exception as e:
            print("Error in find_neighors_from_json_files():", e)
            continue
    return pending_Ids





def dir_json_to_csv_list():
    utils.dir_jsons_to_list(r"D:\Research\sidewalk_wheelchair\DC_DOMs", saved_name=r'D:\Research\sidewalk_wheelchair\jsons.csv')



def sort_jsons():
    utils.sort_pano_jsons(r'D:\Research\sidewalk_wheelchair\DC_DOMs', saved_path=r'D:\Research\sidewalk_wheelchair')


def down_panoramas():
    shape_file = r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\State_of_Washington_DC.shp'
    AOI = gpd.read_file(shape_file)
    saved_path = r'D:\Research\sidewalk_wheelchair'

    down_panos_in_area(polyon=AOI.iloc[0].geometry, saved_path=saved_path, json=True)
    # pass


if __name__ == '__main__':
    # pending_Ids = collect_links_from_panoramas_mp(r'H:\Research\sidewalk_wheelchair\DC_DOMs')
    # print(len(pending_Ids))
    # utils.save_a_list(pending_Ids, r'H:\Research\sidewalk_wheelchair\pendingIds.txt')

    down_panoramas()
    # merge_measurements()
    # dir_json_to_csv_list()
    # sort_jsons()
    # download_panos_DC()
    # get_DOMs()
    # quick_DOM()