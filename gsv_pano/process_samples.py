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
import shutil
from PIL import Image

from tqdm import tqdm

import datetime
import  utils
import fiona
import json
import geopandas as gpd
import pandas as pd
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

def download_pano(latlon_list, saved_path):
    while len(latlon_list) > 0:
        lat, lon = latlon_list.pop(0)
        try:
            pano = GSV_pano(request_lat=lat, request_lon=lon, saved_path=saved_path)
        except Exception as e:
            print("Error in download_pano():", e)

def panorama_from_point_shapefile():
    # shp_file = r''
    print("Reading shaple file...")
    shape_file = r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\noise_map\vectors\SC_road_10m_pt.shp'
    pt_gdf = gpd.read_file(shape_file).to_crs("EPSG:4326")
    pt_gdf['POINT_Y'] = pt_gdf['geometry'].centroid.y
    pt_gdf['POINT_X'] = pt_gdf['geometry'].centroid.x

    print("Making a multiprocess list...")
    lat_lon_mp = mp.Manager().list()
    for idx, row in tqdm(pt_gdf.iterrows()):
        lat = row['POINT_Y']
        lon = row['POINT_X']
        lat_lon_mp.append((lat, lon))


    # AOI = AOI.set_crs("EPSG:2278")
    # pt_gdf = pt_gdf.to_crs("EPSG:4326")
    # pt_gdf['x'] = pt_gdf['geometry'].centroid.x
    # pt_gdf['y'] = pt_gdf['geometry'].centroid.y
    saved_path = r'H:\Research\Noise_map\panoramas4'


    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    csv_name = os.path.join(os.path.dirname(saved_path), os.path.basename(saved_path) + '.csv')

    print("Starting multiprocess...")
    process_cnt = 10
    pool = mp.Pool(processes=process_cnt)
    for i in range(process_cnt):
        pool.apply_async(download_pano, args=(lat_lon_mp, saved_path))
    pool.close()
    pool.join()


    # for idx, row in tqdm(pt_gdf.iterrows()):
    #     lat = row['POINT_Y']
    #     lon = row['POINT_X']
    #     pano = GSV_pano(request_lat=lat, request_lon=lon, saved_path=saved_path)
    print("Saving csv file...")
    dir_json_to_csv_list(saved_path, csv_name)

    print("Saving shaple file...")
    isSave_shp = True
    if isSave_shp:
        shp_name = csv_name.replace('.csv', '.shp')
        print("Saving shapefile: ", shp_name)
        df = pd.read_csv(csv_name)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']))
        gdf.to_file(shp_name)

    print("Done.")

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

def downoad_panoramas_from_json_list(json_file_list, saved_path, zoom=3):
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
    saved_path = r'H:\Research\Noise_map\panoramas4_jpg'
    json_files_path = r'H:\Research\Noise_map\panoramas4'

    # json_files = glob.glob(os.path.join(json_files_path, "*.json"))
    zoom = 2

    gdf = gpd.read_file(r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\noise_map\vectors\panoramas4_max_noise_179k.shp')

    json_files = gdf['panoId'].to_list()
    # panoIds = [os.path.basename(f)[:-5] for f in json_files]

    # downoad_panoramas_from_json_list(json_files, saved_path, zoom)

    logger.info("Making mp_list...")
    panoIds_mp = mp.Manager().list()

    # skips = 0
    for f in tqdm(json_files[:]):
        panoIds_mp.append(os.path.join(json_files_path, f + '.json'))


    process_cnt = 6

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

    process_cnt = 18
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


    print(f"Generated {len(in_points)} seed points inside polygon.bounds {polyon.bounds}")

    print("Start...")
    # random.shuffle(seed_points_mp)

    pending_panoId_mp = mp.Manager().list()

    pending_panoId = collect_links_from_panoramas_mp(saved_path)

    for p in pending_panoId:
        pending_panoId_mp.append(p)

    # download_panoramas_from_seed_points(seed_points_mp, pending_panoId_mp, saved_path, polyon, math.inf, True)

    pool = mp.Pool(processes=process_cnt)

    print(f'Started to download panoramas using {len(in_points)} seed points.')
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

    verbose = True

    seed_points = seed_points[79:]

    while len(seed_points) > 1:

        print(f"Processed {total_cnt - len(seed_points)} / {total_cnt} seed points.")

        seed_point = seed_points.pop()

        lon = seed_point[0]
        lat = seed_point[1]
        pano1 = GSV_pano(request_lon=lon, request_lat=lat)   # DON'T save!

        if pano1.panoId == 0:   # Find no panorama
            continue
        else:
            print(f"Found a panorama in seed point: {seed_point}")

        pending_panoIds.append(pano1.panoId)   # add a panoId to pending_panoIds

        step = 0
        # Divide pending_panoIds and seed_points when thinking. The purpose of seed_points is to initiate and infill the pending_panoIds
        while len(pending_panoIds) > 0:
            try:
                panoId = pending_panoIds.pop()
                json_name = os.path.join(saved_path, panoId + ".json")
                if os.path.exists(json_name):  # This panorama have been downloaded. Skip it.
                    continue
                else:
                    pano2 = GSV_pano(panoId=panoId)

                lon = pano2.lon
                lat = pano2.lat
                pt = Point(lon, lat)
                if pt.within(polygon) and (step < max_step):
                    if not os.path.exists(json_name):
                        with open(json_name, 'w') as f:
                            json.dump(pano2.jdata, f)
                    step += 1

                    links = pano2.jdata["Links"]
                    for link in links:
                        link_panoId = link['panoId']
                        if link_panoId in pending_panoIds:  # move this adjacent panoId to the next.
                            try:
                                pending_panoIds.remove(link_panoId)
                            except:
                                print(f"Failed when removing {link_panoId} in pending_panoIds.")
                        pending_panoIds.append(link_panoId)

                    if step % 100 == 0:
                        print(f"Process (PID) {os.getpid()} has walked {step} steps. Pending panoIds: {len(pending_panoIds)}")
                else:
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
    #shape_file = r'G:\Research\Noise_map\Columbia_MSA_all.shp'
    shape_file = r'H:\My Drive\Research\Charleston_sidewalk\vectors\c_22mr22\charleston.shp'

    AOI = gpd.read_file(shape_file)
    # AOI = AOI.set_crs("EPSG:2278")
    AOI = AOI.to_crs("EPSG:4326")
    saved_path = r'K:\Research\Charleston_sidewalk\pano_json'

    csv_name = os.path.join(os.path.dirname(saved_path), os.path.basename(saved_path) + '.csv')


    for i in range(len(AOI)):
        polygon =  AOI.iloc[i].geometry
        down_panos_in_area(polyon=polygon, saved_path=saved_path, json=True, process_cnt=10, col_cnt=500, row_cnt=500)
    # pass

    dir_json_to_csv_list(saved_path, csv_name)
    isSave_shp = True
    if isSave_shp:
        shp_name = csv_name.replace('.csv', '.shp')
        print("Saving shapefile: ", shp_name)
        df = pd.read_csv(csv_name)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']))
        gdf.to_file(shp_name)

    print("Done.")

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

def get_pano_jpgs(lon_lat_list=[], saved_path=os.getcwd()):
    if len(lon_lat_list) == 0:
        lon_lat_list = [(-94.8696048, 29.2582707)]
    while len(lon_lat_list) > 0:
        lon, lat = lon_lat_list.pop(0)
        pano = GSV_pano(request_lon=lon, request_lat=lat, saved_path=saved_path)
        pano.get_panorama(zoom=5 )



def get_around_thumnail_Columbia():
    saved_path = r'K:\Research\Noise_map\thumnails176k'
    # pano_dir = r'G:\Research\Noise_map'
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    # csv_file = r'G:\Research\Noise_map\panoramas2.csv'
    # df = pd.read_csv(csv_file)
    print("Start to read files...")
    gdf = gpd.read_file(r'K:\OneDrive_USC\OneDrive - University of South Carolina\Research\noise_map\vectors\panoramas4_max_noise_179k_pano_yaw.shp')
    # csv_file = r'G:\Research\Noise_map\panoramas3.csv'
    # df = pd.read_csv(csv_file)
    # df = pd.read_csv(csv_file).sample(frac=1).reset_index()
    gdf = gdf.sample(frac=1).reset_index()
    df = gdf
    start_idx = 0  #+ 60000 #+ 70000  + 70000
    end_idx = len(df)  # 60000 #+ 60000 +   70000 #+  #70000

    print("Start to read download...")
    for idx, row in gdf[:].iterrows():
    # for idx, row in df[:].iterrows():

        panoId = row['panoId']
        pano_yaw_deg = row["pano_yaw_d"]
        pano_yaw_deg = float(pano_yaw_deg)

        bearing_list = [0.0, 90.0, 180.0, 270.0]
        bearing_list = [pano_yaw_deg + b for b in bearing_list]

        print(f"idx: {idx} / {len(df)}, start_idx: {start_idx}, end_idx: {end_idx}, panoId: {panoId}. ")

        # setup_logging(yaml_path, logName=csv_file.replace(".csv", "_info.log"))

        try:

            utils.get_around_thumnail_from_bearing(
                                             panoId=panoId,
                                             bearing_list=bearing_list,
                                             saved_path=saved_path,
                                                   prefix=panoId,
                                                   suffix=['F', 'R', 'B', 'L'],
                                             width=768, height=768,
                                             pitch=0, fov=90,
                                             overwrite=False)

            # print(f"Processing: {idx},  \n")

        except Exception as e:
            print("Error in get_around_thumnail_Columbia, panoId, log:", idx, e)
            continue

def movefiles():
    from_path = r'H:\Research\Noise_map\panoramas4_jpg_half'
    to_path = r'\\DESKTOP-0QRHJF4\Noise_map2\panoramas4_jpg_half'
    to_path = r'K:\Research\Noise_map\panoramas4_jpg_half'

    img_dir = r'\\DESKTOP-H2OGE6L\panoramas4_jpg_half\*.jpg'

    jpgs = glob.glob(img_dir)

    new_jpgs = glob.glob(os.path.join(to_path, "*.jpg"))

    for j in new_jpgs:
        try:
            basename = os.path.basename(j)
            new_name = os.path.join(img_dir[:-6], basename)
            jpgs.remove(new_name)
        except Exception as e:
            print(e, j, new_name)
    # jpgs = jpgs[::-1]
    print(f"Image count: {len(jpgs)}")
    skip_cnt = 0
    copied_cnt = 0
    for idx, jpg in tqdm(enumerate(jpgs[:])):


        basename = os.path.basename(jpg)
        new_name = os.path.join(to_path, basename)

        if idx % 1000 == 0:
            print(f"Processed {idx} image. skip_cnt: {skip_cnt}, copied_cnt: {copied_cnt}")

        if os.path.exists(new_name):
            skip_cnt = skip_cnt + 1
            continue

        shutil.copyfile(jpg, new_name)
        copied_cnt += 1
        # cmd = f"copy {jpg} {new_name}"
        # os.system(cmd)
        if idx % 1000 == 0:
            print(f"Processed {idx} image. skip_cnt: {skip_cnt}, copied_cnt: {copied_cnt}")

def test_get_depthmap():
    saved_path = os.getcwd()

    print(Image.__version__)

    lon = -77.072465
    lat = 38.985399

    pano1 = GSV_pano(request_lon=lon, request_lat=lat, crs_local=6487, saved_path=saved_path)
    pano1.get_depthmap(zoom=0, saved_path=saved_path)
    pano1.get_DOM(resolution=0.1)


if __name__ == '__main__':
    # pending_Ids = collect_links_from_panoramas_mp(r'H:\Research\sidewalk_wheelchair\DC_DOMs')
    # print(len(pending_Ids))
    # utils.save_a_list(pending_Ids, r'H:\Research\sidewalk_wheelchair\pendingIds.txt')

    # draw_panorama_apex_mp(saved_path=r"H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_panoramas\sidewalk_wheelchair",
    #                    json_dir=r'H:\Research\sidewalk_wheelchair\DC_DOMs')

    # download_panoramas()
    # panorama_from_point_shapefile()
    # merge_measurements()
    # dir_json_to_csv_list(json_dir=r'D:\Research\sidewalk_wheelchair\jsons', saved_name=r'D:\Research\sidewalk_wheelchair\jsons250k.txt')
    # sort_jsons()
    # download_panos_DC()
    # download_panos_DC_from_jsons()
    # get_DOMs()
    # get_pano_jpgs()
    # get_DOMs()
    # get_around_thumnail_Columbia()
    # quick_DOM()
    # movefiles()
    test_get_depthmap()