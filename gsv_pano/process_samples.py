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

import pandas as pd



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

def get_panorama(mp_list, saved_path, zoom=4, json_only=True):
    total = len(mp_list)
    saved_path = saved_path
    processed_cnt = 0
    while len(mp_list) > 0:
        try:
            i, lon, lat = mp_list.pop(0)
            pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=saved_path)

            if not json_only:
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
            geometry = points[i].shape.__geo_interface__['coordinates']  # using pyshp

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
                    try:
                        seg_files.remove(temp_name)
                        seg_files.append(temp_name)
                    except:
                        pass

            if is_processed:
                print("Skip: ", seg_file)
                continue
            pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=saved_path)
            # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=r'J:\Research\StreetView\gsv_pano\test_results')
            pano1.set_segmentation_path(full_path=seg_file)
            DOM = pano1.get_DOM(width=40, height=40, resolution=resolution, zoom=4, img_type='segmentation',  fill_clipped_seg=True)
            total_time = (time.perf_counter() - start_time_all)

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

    process_cnt = 8
    pool = mp.Pool(processes=process_cnt)

    for i in range(process_cnt):
        pid_id = i
        pool.apply_async(get_DOM, args=(pid_id, seg_files_mp, saved_path, resolution))
    pool.close()
    pool.join()



    pool.join()

def quick_DOM():

    seg_dir = r'D:\DC_segmented'
    pano_dir = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_panoramas'
    seg_files = glob.glob(os.path.join(seg_dir, "*.png"))
    pano_files = glob.glob(os.path.join(pano_dir, "*.jpg"))
    saved_path = r'F:\Research\sidewalk_wheelchair\DOMs_full'
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
            pano1.depthmap['ground_mask'][int(mask_h / 4 * 4):, :] = 0



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
    saved_file0 = r'D:\Research\sidewalk_wheelchair\widths_all4_not_touched.txt'
    saved_file1 = r'D:\Research\sidewalk_wheelchair\widths_all4_touched.txt'
    widths_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2'
    sorted_panoIds = open(sorted_file, 'r').readlines()
    sorted_panoIds = [x[:-1] for x in sorted_panoIds][:]

    header = r'panoId,contour_num,center_x,center_y,length,col,row,end_x,end_y,cover_ratio,is_touched'

    touched_lines = []
    not_touched_lines = []

    two_lists = [not_touched_lines, touched_lines]

    cover_ratio_threshold = 0.85

    for idx, panoId in enumerate(sorted_panoIds):
        try:

            file_name = os.path.join(widths_dir, f'{panoId}_widths.csv')
            if not os.path.exists(file_name):
                print("No width measurement!")
                continue
            if idx % 100 == 0:
                print(f'Processed {idx} / {len(sorted_panoIds)}')

            lines = open(file_name, 'r').readlines()[1:]
            for line in lines:
                fields = line.split(',')
                cover_ratio = float(fields[9])
                is_touched = int(line[-2])
                if (cover_ratio > cover_ratio_threshold) and (is_touched == 0):
                    two_lists[0].append(line)
                else:
                    two_lists[1].append(line)


        except Exception as e:
            print("Error in merge_measurements:", e)
            continue

    print("Writing: ", saved_file0)

    with open(saved_file0, 'w') as f:
        f.writelines(header + '\n')
        f.writelines(''.join(two_lists[0]))

    print("Writing: ", saved_file1)
    with open(saved_file1, 'w') as f:
        f.writelines(header + '\n')
        f.writelines(''.join(two_lists[1]))

    print("Done.")

def down_panos_in_area(polyon, saved_path='', col_cnt=100, row_cnt=100, json=True, pano=False, pano_zoom=0, depthmap=False, process_cnt=10):
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
    print(f"Generate {len(in_points)} seed points inside polygon.bounds {polyon.bounds}")

    random.shuffle(seed_points_mp)

    pending_panoId_mp = mp.Manager().list()

    pending_panoId = []
    pending_panoId = collect_links_from_panoramas_mp(saved_path)   # the existing jsons.



    for p in pending_panoId:
        pending_panoId_mp.append(p)

    if process_cnt == 1:
        download_panoramas_from_seed_points(
        seed_points_mp, pending_panoId_mp, saved_path, polyon, math.inf, True)
    # download_panoramas_from_seed_points(seed_points_mp, pending_panoId_mp, saved_path, polyon, math.inf, True)

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

    while len(seed_points) > 0:

        print(f"Processed {total_cnt - len(seed_points)} / {total_cnt} seed points.")

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
                    #pano2 = GSV_pano(json_file=json_name)
                    # pano2 = GSV_pano(panoId=panoId)
                    points_name = os.path.join(saved_path, panoId + ".npy")  # 52 m per file for zoom=2
                    if not os.path.exists(points_name):
                        g_points = pano1.get_ground_points(zoom=0, color=True)
                        np.save(points_name, g_points)  # 55 m per file
                        pano1.get_panorama(saved_path=saved_path, zoom=3)
                    continue
                else:
                    pano2 = GSV_pano(panoId=panoId)


                    # print("Downloaded: ", pano2.panoId)


                lon = pano2.lon
                lat = pano2.lat
                pt = Point(lon, lat)
                if pt.within(polygon) and (step < max_step):
                    with open(json_name, 'w') as f:
                        json.dump(pano2.jdata, f)
                        g_points = pano2.get_ground_points(zoom=0, color=True)
                        #points_name = os.path.join(saved_path, panoId + ".npz")
                        #np.savez(points_name, ground_points=g_points)   # 55 m per file for zoom=2
                        points_name = os.path.join(saved_path, panoId + ".npy")   # 52 m per file for zoom=2
                        np.save(points_name, g_points)  # 55 m per file
                        #with open(points_name, 'w'
                        #print(g_points)
                        # downloaded_cnt += 1
                    step += 1
                    if step % 5 == 0:
                        print(f"Process (PID) {os.getpid()} has walked {step} steps.")

                    links = pano2.jdata["Links"]
                    for link in links:
                        link_panoId = link['panoId']
                        if link_panoId in pending_panoIds:
                            try:
                                pending_panoIds.remove(link_panoId)

                            except:
                                pass
                        pending_panoIds.append(link_panoId)
                        print('link_panoId:', link_panoId)
                    print(f"Processed: {panoId}. Len of pending_panoIds: {len(pending_panoIds)}")
                else:
                    print(f"{panoId} is outside the area of interest.")
                    continue

            except Exception as e:
                print("Error in download_panoramas_from_seed_point():", e)
                continue

        print(f"Downloaded {step} panoramas for this seed point.")

def collect_links_from_panoramas_mp(json_dir, process_cnt=10):
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





def dir_json_to_csv_list(json_dir, saved_name):
    utils.dir_jsons_to_list(json_dir, saved_name)



def sort_jsons():
    utils.sort_pano_jsons(r'D:\Research\sidewalk_wheelchair\DC_DOMs', saved_path=r'D:\Research\sidewalk_wheelchair')


def download_panoramas():
    # shape_file = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\State_of_Washington_DC.shp'
    # AOI = gpd.read_file(shape_file)
    # saved_path = r'D:\Research\sidewalk_wheelchair\jsons'
    saved_path = r'H:\Research\interchange\high_five\jsons'

    # polygon = shapely.geometry.box(minx=-85.7499798, miny=38.2547416, maxx=-85.7337196, maxy=38.2677969)  # Kennedy interchange, Louisville, KY
    polygon = shapely.geometry.box(minx=-96.7732, miny=32.9193, maxx=-96.7559, maxy=32.9301)  # Dallas High FIve Intercahnge, Dallas, TX.

    # down_panos_in_area(polyon=AOI.iloc[0].geometry, saved_path=saved_path, json=True, process_cnt=4)
    down_panos_in_area(polyon=polygon, saved_path=saved_path, json=True, process_cnt=4)

    dir_json_to_csv_list(json_dir=saved_path, saved_name=r'H:\Research\interchange\high_five\high_five_jsons.csv')
    # pass

def draw_panorama_apex_mp(json_dir='', saved_path='', local_crs=6487, process_cnt=10):

    print("Start to collect json files...")
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    panoIds = [os.path.basename(f)[:-5] for f in json_files]

    print(f"Start to process {len(panoIds)} files...")
    results_mp = mp.Manager().list()


    # draw_panorama_apex(panoIds[0:20], json_dir, saved_path, results_mp, local_crs)

    panoIds_mp = mp.Manager().list()
    for p in panoIds[:]:
        panoIds_mp.append(p)



    pool = mp.Pool(processes=process_cnt)
    for i in range(process_cnt):
        pool.apply_async(draw_panorama_apex, args=(panoIds_mp, json_dir, saved_path, results_mp, local_crs))
    pool.close()
    pool.join()

    utils.save_a_list(results_mp, r'D:\Research\sidewalk_wheelchair\degrees.txt')



def draw_panorama_apex(panoIds, json_dir, saved_path, results, local_crs=6487):
    print("PID: ", os.getpid())
    total_cnt = len(panoIds)
    print("total_cnt:", total_cnt)
    processed_cnt = 0

    transformer = utils.epsg_transform(in_epsg=4326, out_epsg=local_crs)

    while len(panoIds) > 0:
        panoId = panoIds.pop()

        processed_cnt += 1
        if len(results) % 100 == 0:
            print(f"Processed {len(results)} jsons.")

        json_file = os.path.join(json_dir, panoId + ".json")
        if os.path.exists(json_file):

            local_crs = 6487
            # print(json_file)
            pano = GSV_pano(json_file=json_file)


            Links = pano.jdata["Links"]
            if len(Links) < 2:
                # print(f"Error in draw_panorama_apex(): {pano.panoId} has no 2 panoramas in Links.")
                continue

            try:

                json_file_0 = os.path.join(json_dir, Links[0]['panoId'] + ".json")
                json_file_1 = os.path.join(json_dir, Links[1]['panoId'] + ".json")

                # print("json_file_0:", json_file_0)

                pano_0 = GSV_pano(json_file=json_file_0)
                pano_1 = GSV_pano(json_file=json_file_1)

                # print("pano_1.panoId:", pano_1.panoId)

                if (pano_1.panoId == 0) or (pano_0.panoId == 0):
                    # print("Error in Links:")
                    continue
                # pano_1 = GSV_pano(panoId=Links[1]['panoId'])

                # print("Line 532")

                # print("Line 572")
                xy = transformer.transform(pano.lat, pano.lon)
                xy0 = transformer.transform(pano_0.lat, pano_0.lon)
                xy1 = transformer.transform(pano_1.lat, pano_1.lon)
                pts = np.array([xy0, xy, xy1])
                # print("Line 577")

                # calculate angle
                a = (xy[1] - xy0[1], xy[0] - xy0[0])
                a = np.array(a)
                b = (xy[1] - xy1[1], xy[0] - xy1[0])
                b = np.array(b)
                angle = np.arccos(np.dot(a, b) / (LA.norm(a) * LA.norm(b)))
                angle_deg = np.degrees(angle)
                # print("Line 540")

                draw_fig = False

                results.append(f"{pano.panoId},{angle_deg:.2f}")

                if draw_fig:
                    max_x = pts[:, 0].max()
                    max_y = pts[:, 1].max()
                    min_x = pts[:, 0].min()
                    min_y = pts[:, 1].min()

                    range_x = max_x - min_x
                    range_y = max_y - min_y

                    range_max = max(range_y, range_x) * 1.5
                    x_center = (max_x + min_x) /2
                    y_center = (max_y + min_y) /2

                    plt.axis('scaled')
                    # plt.axis("off")


                    plt.xlim(x_center - range_max /2, x_center + range_max /2)
                    plt.ylim(y_center - range_max /2, y_center + range_max /2)

                    plt.plot(pts[:, 0], pts[:, 1])
                    plt.scatter(pts[:, 0], pts[:, 1], marker='o', color='red')
                    anno_texts = [pano_0.panoId, pano.panoId, pano_1.panoId]


                    texts = []
                    for i, txt in enumerate(anno_texts):
                        texts.append(plt.annotate(txt, pts[i], ha='center'))
                    # plt.axis('square')

                    adjust_text(texts, only_move={'points': 'y', 'texts': 'y'},
                                expand_text=(1.25, 1.3),
                                expand_objects=(1.25, 1.4),
                                expand_align=(1.25, 1.4),
                                arrowprops=dict(arrowstyle="->", color='r', lw=0.5))


                    # print(f"angle: {angle_deg:.2f}")
                    # plt.show()
                    ax = plt.gca()
                    # plt.title(f"{pano.panoId}: {angle_deg:.2f}")
                    ax.xaxis.set_label_position('top')
                    ax.set_title(f"{pano.panoId}: {angle_deg:.2f}", y=1.0, pad=-14)
                    plt.savefig(os.path.join(saved_path, panoId + '.png'))

            except Exception as e:
                print(f"Error in draw_panorama_apex():", e, json_file)
                continue


def download_panos_Columbia():
        logger.info("Started...")
        saved_path = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\Columbia_LEF\jsons'
        shp_path = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\Columbia_LEF\vectors\kx-south-carolina-address-points-SHP\columbia_metro.shp'
        points = fiona.open(shp_path)

        print("Len of shp:", len(points))

        skips = 0

        # points = points[skips:]

        print("Len of shp after skipping:", len(points))

        logger.info("Making mp_list...")
        lonlats_mp = mp.Manager().list()

        for i in range(len(points) - skips):
            i += skips
            geometry = points[i]['geometry']['coordinates'] # using fiona
            # geometry = points[i].shape.__geo_interface__['coordinates']  # using pyshp

            lon, lat = geometry
            lonlats_mp.append((i, lon, lat))
        logger.info("Finished mp_list (%d records).", len(lonlats_mp))

        cut_point = 100
        lonlats_mp_first100 = lonlats_mp[:cut_point]
        random.shuffle(lonlats_mp_first100)
        # lonlats_mp[:cut_point] = lonlats_mp_first100

        random.shuffle(lonlats_mp)

        process_cnt = 8
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


def getHouse_image_Columbia():
    saved_path = r'H:\Research\Columbia_LFE\Google_thumbnails30'
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    csv_file = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\Columbia_LEF\vectors\kx-south-carolina-address-points-SHP/columbia_metro.csv'
    df = pd.read_csv(csv_file)

    for idx, row in df[108593:].iterrows():

        ID = int(row['ID'])
        h_LAT = row['LAT']   # h_ means heading
        h_LON = row['LON']

        print("idx, ID:  ", idx, ID)

        setup_logging(yaml_path, logName=csv_file.replace(".csv", "_info.log"))



        logger.info("ID: %s polygon.centroid: %f, %f", ID, h_LON, h_LAT)

        # panoId, pano_lon, pano_lat = gpano.getPanoIDfrmLonlat(h_LON, h_LAT)



        try:

            # jdata = gpano.getPanoJsonfrmLonat(h_LON, h_LAT)
            gpano = GSV_pano(request_lon=h_LON, request_lat=h_LAT)
            jdata = gpano.jdata
            panoid = jdata['Location']['panoId']
            pano_lon = float(jdata['Location']['lng'])
            pano_lat = float(jdata['Location']['lat'])
            pano_yaw = float(jdata['Projection']['pano_yaw_deg'])

            heading = gpano.getDegreeOfTwoLonlat(pano_lat, pano_lon, h_LAT, h_LON)

            # determine toward to right or left
            yaw_r = (pano_yaw + 90) % 360
            yaw_l = (pano_yaw + 270) % 360
            diff = [heading - yaw_r, heading - yaw_l]
            # diff
            diff_cos = [math.cos(math.radians(diff[0])), math.cos(math.radians(diff[1]))]
            max_idx = np.argmax(diff_cos)
            yaw = [yaw_r, yaw_l][max_idx]

            fov = 30

            gpano.getImagefrmAngle(pano_lon, pano_lat, saved_path=saved_path,
                                   prefix=ID, yaw=yaw, fov=fov, height=768, width=768)

            gpano.getImagefrmAngle(pano_lon, pano_lat, saved_path=saved_path,
                                   prefix=ID, yaw=yaw-30, fov=fov, height=768, width=768)

            gpano.getImagefrmAngle(pano_lon, pano_lat, saved_path=saved_path,
                                   prefix=ID, yaw=yaw+30, fov=fov, height=768, width=768)

            gpano.getImagefrmAngle(pano_lon, pano_lat, saved_path=saved_path,
                                   prefix=ID, yaw=yaw-15, fov=fov, height=768, width=768)

            gpano.getImagefrmAngle(pano_lon, pano_lat, saved_path=saved_path,
                                   prefix=ID, yaw=yaw+15, fov=fov, height=768, width=768)


            print(f"Processing: {idx},  \n")

        except Exception as e:
            print("Error in getHouse_image_HamptonRoads, panoId, log:", idx, e)
            continue



def getHouse_image_Columbia2():
    saved_path = r'H:\Research\Columbia_LFE\Google_thumbnails30_2'
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    csv_file = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\Columbia_LEF\vectors\kx-south-carolina-address-points-SHP/columbia_metro.csv'

    json_dir = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\Columbia_LEF\all_jsons'

    json_files = glob.glob(os.path.join(json_dir, '*.json'))

    print(f"Found {len(json_files)} files in: {json_dir}")

    random.shuffle(json_files)

    # df = pd.read_csv(csv_file)
    for idx, f in enumerate(json_files):
    # for idx, row in df[0:].iterrows():

        # ID = int(row['ID'])
        # h_LAT = row['LAT']   # h_ means heading
        # h_LON = row['LON']

        # print("idx, ID:  ", idx, ID)

        setup_logging(yaml_path, logName=csv_file.replace(".csv", "_info_columbia_pano_download.log"))

        # logger.info("ID: %s polygon.centroid: %f, %f", ID, h_LON, h_LAT)

        # panoId, pano_lon, pano_lat = gpano.getPanoIDfrmLonlat(h_LON, h_LAT)


        try:

            # jdata = gpano.getPanoJsonfrmLonat(h_LON, h_LAT)
            # gpano = GSV_pano(request_lon=h_LON, request_lat=h_LAT)
            gpano = GSV_pano(json_file=f)
            jdata = gpano.jdata
            panoid = jdata['Location']['panoId']
            pano_lon = float(jdata['Location']['lng'])
            pano_lat = float(jdata['Location']['lat'])
            pano_yaw = float(jdata['Projection']['pano_yaw_deg'])

            # heading = gpano.getDegreeOfTwoLonlat(pano_lat, pano_lon, h_LAT, h_LON)
            #
            # # determine toward to right or left
            # yaw_r = (pano_yaw + 90) % 360
            # yaw_l = (pano_yaw + 270) % 360
            # diff = [heading - yaw_r, heading - yaw_l]
            # # diff
            # diff_cos = [math.cos(math.radians(diff[0])), math.cos(math.radians(diff[1]))]
            # max_idx = np.argmax(diff_cos)
            # yaw = [yaw_r, yaw_l][max_idx]

            fov = 30
            phi_list = [60, 75, 90, 105, 120, -60, -75, -90, -105, -120]
            gpano.get_image_from_headings(saved_path=saved_path, phi_list=phi_list, fov=fov, height=768, width=768, override=False)

            if idx % 1 == 0:
                print(f"Processing: {idx} / {len(json_files)}. ")

        except Exception as e:
            print("Error in getHouse_image_Columbia2, panoId, log:", idx, e)
            continue
if __name__ == '__main__':
    # pending_Ids = collect_links_from_panoramas_mp(r'H:\Research\sidewalk_wheelchair\DC_DOMs')
    # print(len(pending_Ids))
    # utils.save_a_list(pending_Ids, r'H:\Research\sidewalk_wheelchair\pendingIds.txt')

    # draw_panorama_apex_mp(saved_path=r"H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_panoramas\sidewalk_wheelchair",
    #                    json_dir=r'H:\Research\sidewalk_wheelchair\DC_DOMs')

    # download_panoramas()

    # merge_measurements()
    # dir_json_to_csv_list(json_dir=r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\Columbia_LEF\all_jsons', saved_name=r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\Columbia_LEF\jsons760k.txt')
    # sort_jsons()
    # download_panos_DC()
    # download_panos_DC_from_jsons()
    # get_DOMs()
    # get_DOMs()
    # quick_DOM()
    # download_panos_Columbia()
    getHouse_image_Columbia2()