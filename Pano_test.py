import GPano
import json
import os
import requests
import pandas as pd
import multiprocessing as mp
import time

def getJsonDepthmapfrmLonlat(lon, lat, dm=1, saved_path='', prefix='', suffix=''):
    prefix = str(prefix)
    suffix = str(suffix)
    if prefix != "":
        prefix += '_'
    if suffix != "":
        suffix = '_' + suffix
    url = f'http://maps.google.com/cbk?output=json&ll={lat},{lon}&dm={dm}'
    # print(url)
    try:
        r = requests.get(url)
        jdata = r.json()
        # str_dm = jdata['model']['depth_map']

        mapname = os.path.join(saved_path, prefix + jdata['Location']['original_lng'] + '_' + jdata['Location'][
            'original_lat'] + '_' + jdata['Location']['panoId'] + suffix + '.json')

        with open(mapname, 'w') as f:
            json.dump(jdata, f)

    except Exception as e:
        print("Error in getPanoIdDepthmapfrmLonlat():", str(e))
        print(url)

def getJsonDepthmapsfrmLonlats(lonlat_list, dm=1, saved_path='', prefix='', suffix=''):
    start_time = time.time()
    Cnt = 0
    Cnt_interval = 1000
    origin_len = len(lonlat_list)

    while len(lonlat_list) > 0:
        lon, lat, id, idx = lonlat_list.pop(0)
        prefix = id
        prefix = str(prefix)
        suffix = str(suffix)
        if prefix != "":
            prefix += '_'
        if suffix != "":
            suffix = '_' + suffix
        url = f'http://maps.google.com/cbk?output=json&ll={lat},{lon}&dm={dm}'
        print("Current row:", idx)
        try:
            r = requests.get(url)
            jdata = r.json()
            # str_dm = jdata['model']['depth_map']

            mapname = os.path.join(saved_path, prefix + jdata['Location']['original_lng'] + '_' + jdata['Location'][
                'original_lat'] + '_' + jdata['Location']['panoId'] + suffix + '.json')

            with open(mapname, 'w') as f:
                json.dump(jdata, f)

            current_len = len(lonlat_list)
            Cnt = origin_len - current_len
            if Cnt % Cnt_interval == (Cnt_interval - 1):
                print(
                    "Prcessed {} / {} items. Processing speed: {} points / hour.".format(Cnt, origin_len, int(
                        Cnt / (time.time() - start_time + 0.001) * 3600)))


        except Exception as e:
            print("Error in getJsonDepthmapsfrmLonlats():", str(e))
            print(url)
            continue

def getJsonDepthmapsfrmLonlats_mp(lonlat_list, dm=1, saved_path='', prefix='', suffix='', Process_cnt=4):

    try:
        pool = mp.Pool(processes=Process_cnt)
        for i in range(Process_cnt):
            pool.apply_async(getJsonDepthmapsfrmLonlats, args=(lonlat_list, dm, saved_path, prefix, suffix))
        pool.close()
        pool.join()


    except Exception as e:
        print("Error in getJsonDepthmapsfrmLonlats_mp():", str(e))

if __name__ == '__main__':
    # getJsonDepthmapfrmLonlat(-74.317149454999935, 40.798423060000061, dm=1,
    #                          saved_path=r'J:\Sidewalk')

    saved_path = r'J:\Sidewalk\google_street_view\Essex_detpthmap'
    #saved_path = r'J:\Sidewalk\t'
    df = pd.read_csv(r'K:\Research\NJTPA\Essex_10m_points.csv')
    df = df.fillna(0)
    #df = df[20000:]
    print('len of df:', len(df))
    mp_list = mp.Manager().list()
    for idx, row in df.iterrows():
        mp_list.append([row.lon, row.lat, row.id, str(int(idx))])
    print('len of mp_list:', len(mp_list))
    print('Started!')
    try:
        getJsonDepthmapsfrmLonlats_mp(mp_list, dm=1, saved_path=saved_path, Process_cnt=6)

    except Exception as e:
        print("Error in row: ",  e)
