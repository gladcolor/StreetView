import GPano
import json
import os
import requests

import multiprocessing as mp
import time
import GPano
import glob
import math

import PIL
from PIL import Image, features

import pandas as pd
import numpy as np
from natsort import natsorted
from pyproj import CRS
from pyproj import Transformer


import pptk

gsv = GPano.GSV_depthmap()
gpano = GPano.GPano()

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


def test_gsv_parse():
    s = 'eJzt2AlwFvUdxvHc70tCMMRAQjgk3IkpRUsg5ODdN0pUHESGoRapLTNApVUKrS1HSbKDrUYYjtJ6QBFECKIcsYgwxHffrO2oVSoI9aLA0EIpkVIVBOvRAn3fNwdvkvfY3Xd3f__d__OdTGYyyYTN7_PkGJylcXEJcfHOOIQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhNpKaBf10yCjS1AV9dMivVLnjh3YJkeg2PgxA0vmuJY-_hiBZXK0T0d_bID1HJ3T2R8bYLUQ9gb5YwLMFQbfMH-rTiBeSdQPqbbw-Eb6W2kCitgtuoOI-sb6W2AC2uQtM4Mo-Mb7M7wAHegZH0F0feP52VyArvaMbkCBvjn-jE3AEHvmNqBI3zR_VhZgqD1DG1Cob6I_AwNwmqLPwAQU65vpT7wApy_z_EkXoILfVH_CBTidZvuTTUCNvtn-NANwtmayP8kE1PGb7U-wAKeT0N_sBajUJ_A3eQBOJ7G_qQtQzU_gb-YCnE4G_M2bgHp-En-zBuDsGJm_KQvQoE_kb8oAOumT-hs_AE38RP7GDyCEPq2_0QvQxk_lb_QAQvIT-xu6AI38ZP6GLiC0Pr2_cQvQyk_ob9wAwvEz4G_QADTzU_obNICw-kz4G7IA7fyk_oYMIAI_G_76DyAGflp__QcQSZ8Vf70HEAu_zfwj87Pir-8AYuIn9td5AFH4mfHXdQCW9td1ANH42fHXcQGx8ZP76ziAqPws-es1AKvz6zaA6Pps-eszgBj5WfDXZwBK-Nny12MACfBvThE_Y_46DMAW_joMQBk_a_4xDyDBHv7dTeJnzj_GASTAXxU__G3pr5ifPf-YBuA_HfxV8NtrAIHT2cM_pgFY21_7AAKng78afhb9tQ6g-XTc-6viZ9Jf2wBaTmcTf4c5_PC3mb9Kfjb9tQyg9XR28dc4AHv4axhA6-ls469pAGr5WfVXPYC203Htr5rfLv7XTmcff_UDUM_PrL_KAVw7Hfzt4a9qAEGns5G_2gFo4Ie_ffy18DPsr2YAQaezk7-qAWjiZ9lfxQCCTgd_-NvGX8UAtPEz7Z8Of_jz7a98AHb0VzyAoNNx6p9mS3-lAwg6nc38lQ0gLQ3-LfHnnxbInv4KBxB0Orv5RxtAWhr84c-pf9eu9vePMgCb-_fsGeFrd3RtKeh08Ie_vfwjDiCDY38H_G3tHw__aAPI4NjfAX_4c-EfdgAZ8Ic_p_4OjvzDDYAL_6yskP8LBH8_v_39s0L7O-DPj3-IAfDlH3IAGfCHPx_-nQbg4Mw_Oww__OHPg396KH6e_G_oyO-EP1f-ffv25dk_PQQ_V_7Z8OfdP_iPACdv_rmm-JORtyukf3_4d-Dn2N8Jf_jz5Z8L_2B--MMf_vDnx98Jf_gH83PnP4Rv_1z4wx_-8Ic__Dn2H_4N-MMf_vCHP_w58x81Cv68-uc3V1RUBH_u_Ivz8wsKEhMTS25siT__MXFxI30vI7nzL2guL69PyIJOZ1__YW31C8SJf15QAwf6XnXrXNDpbOk_wl-_TnHhPzConJzrQ5Tjq-10tvMfPfomf0PbGhxcrP5MDyA9vUX4Zn8-5czMnAgFTmcr_1Br75CN_YNoM1vraD4oOP_p7OOvAD9oBbbyzwxTamq497RmF_8Qf-S01StM9vBPjVg0_8xMO_h_M5Baf00boOZuV2T78AMY0D6L-18XpUg_GwJZ0l-BfWpSUpKSD_NnUf9o9or81WyAmj1QkuKU8mudAPP4Sv27dSsstIK_cnrV_lo2wDx-RP9CfyntY9hfpb0Wf7UbYB4_OTk5RX3M-Sf2vpbR_mo2QGCfrDYN_uFHQEDfIVVL0OqvdAOs28fiH3IEpPSdizyGxMRY_JVsgHX7mP07joAdekXF6h9tA6zb6-MfNAPr0Ovm76sHnX9M9Pr6-7MKvH7-PZoL_EIx1z92ev39jViAAe56-Qfjt2WCv070RvjrtQMD1fXhD2nfYQZsyxvqr3YIZnjr5x8Nv12Mwpvk31aX1ijJdfBXZR_zFAxzp_NvjZpfg38s9irWYDg6_FX760sfyEzmsMGfgh7-1vA3ih7-jPsbCs-Qv2n87PmTucOfKX8zyeHPjr_54PCHP6_-nfjhz8IA6Pyp-eHvD_7whz_84Q9_-MMf_vCHP_zhD3_4wx_-8Ic__OEPf_jDH_7whz_84Q9_-MMf_vCHP_zhD3_4wx_-8Ic__OEP_4j88E9iYQBk_tT68Ic_edT68KeNWh_-tFHrw582an3400atD3_aqPXhTxu1Pvxpo9aHP23U-vCnjVof_rRR68OfNmp9-NNGrQ9_2qj14U8cNb9Z_gzywx_-5FHzw582an7400bND3_aqPnhTxs1P_xpo-aHP23U_PCnjZof_rRR88OfOD78WeSHP_zpgz9Z1PSB4E8WNX0g-JNFTR8I_mRR0weCP1nU9M3x4M8kP_zhz0Dwp4pavjn4U0Ut3xz8qaKWb4lDf2r55qjhW7K_P5vf_vCHPwvBnyhq-Nbs7s8oPzMDgD9R1PAtwZ8oavjW7O3PLD_84c9E8KeJ2r0tO_uzyw9_-DOSff0Z5mfIn24AHPPDn-9vf5b8qQbAMz9T_kQD4JmfLX-aAfDMz5g_yQI41mfPn2AAPPOz52_-AjjWZ9I_yeQJmKNPDR0maumwWZrfIviJDPv7s6S_dez9URNHzzr8lviF3yFqXRUx629B9mtRq2qOkt_K4B2iZjQ0Tfr2sVUWNRJZ1IdnKmoMM6O-NTvFdehn20XX5MsTvYvqXiru_Xi827tJlO-ed2_5GSm5oWr4R0KP3bd5nvjQKf3wsigPPJgpFBzNLe0uHfJUXxHlca9lCul7t9TdPKiPlOJ7_-7-ucLnFas8C3dkS6v_J8onfG-P-pso710eX_7s-YfGdH28Saj_QHSdmtPdO0Ec0VBzW6p72YfjN02YcUl6_5Qo_-jrXsKiLKn04LgLnsG-z1eX1ENYWz_LdU_WqbEPfZEg1dZvcz91d2bj0m9Xy-8U1paNHVRZ8dlZ0XWi-j2p8qfbG5YU_EN4cvN8aWp8nPTb__ierzBHWPD9-z3DHrur9NarojyjarAwaJ3oWjx0hvfKWwVjKg-dFM5JVfLKswMaL714X-mQnCJh8ZGrngPz13ju9P37B57LFKYsaJTyrmZI-86J8up1uULB6_WNXz85Tz67ZUnZulfKKvq8vLxk7vzc8qmHRfn80hTh5S8uln35V1F-a25t2cxuPSs2vCnKO6YNKx85YkPxRxc-ER6Ynuy99chKzzPHRXn7xUzh6OsvSE2HS6RtH4vy-oFZwoRPRdf-0_ukbdKbDZdXvi1smJld-sCVJd4di0W56MESof4HfcoSVtdKqZdEeeauIcLGplljL8xZ5Ip7-hHpvslnKiZdTi1ryj7uKfDd_y_ZxUK_FYLcv36Y64SYL-0ce0fF_Q-L8uj51eW5k1c0nL0r3Z3w61-WrBiR6n3hqCindp8i5C9YVbqx_7-8eyfVyN_LE4XqLQulyoNdvFPfFeVli6cIKQVfeR5-9Y7GxV_9Qv7Tt_YL004MkW7fvV76-XlR_iTdITQur5LvXTZ07KT6OE_tgk_djyZelP57YLX0xhHfPiY6hd8cSix7LOu0tOi0KL8xdKNr65H90oQDmz09ffe9cKmL8OOTvaVj92z1DPc9f9XpNOH4ow9K17ue9sz5UpTdrzqERXN7S3sO7vSuyxDl1_74vGtm5Xulmz8XXYn7aqW6V2685eQJ0fXSyARv2dCfNFQm5bkHl6VJC9ZuKpeHi_KVcY8Ik5-dJ63aXVi-pVGUK27fIdQ4T0nbt-4prztdI5-Z_o4gbWqUirwfeJt-VSOX7_q9IP7hd9Lh_sOkY76vb_a08cKMf35W0iv_-fI1A0R5uvM595In1kvS8XPeBleNvPC6YnfO9I3S2rrveHc95dvvv29y39L9RckxOsX7_gFRHv73P7vEibul8WuWekd8V5R7T4pz72mqkt4-Vlw-e48oV-90uZPfLS6rnbXUWzhbdNV-nCT8H5Kkoqk'
    gpano = GPano.GPano()
    dm = GPano.GSV_depthmap()
    #jdata = gpano.getJsonDetthmapfrmLonlat()

    depthMapData = dm.parse(s)
    print(depthMapData.shape)

def test_getPanoIDfrmLonlat():
    lon = -77.072465
    lat = 38.985399
    gpano = GPano.GPano()
    print(gpano.getPanoIDfrmLonlat(lon, lat))

def test_getJsonfrmPanoID():
    lon = -77.072465
    lat = 38.985399
    gpano = GPano.GPano()
    panoId, pano_lon, pano_lat = gpano.getPanoIDfrmLonlat(lon, lat)
    result = gpano.getJsonfrmPanoID(panoId)
    print(result)
    print(result['Time_machine'])

def test_go_along_road_forward():
    csv_file = r'L:\Datasets\HamptonRoads\EC-lat-lon.csv'
    df = pd.read_csv(csv_file)
    for idx, row in df[3750:].iterrows():
        lon = row['LON']
        lat = row['LAT']
        ID = int(row['ID'])
        gpano = GPano.GPano()
        saved_path = r'L:\Datasets\HamptonRoads\go_along'
        results = gpano.go_along_road_forward(lon, lat, saved_path, yaw_list='json_only', steps=100, overwrite=True)
        # results = gpano.go_along_road_backward(lon, lat, saved_path, yaw_list='json_only', steps=100, overwrite=True)

        # go_along_road_forward(self, lon, lat, saved_path, yaw_list=0, pitch_list=0, steps=99999, polygon=None, fov=90, zoom=5):
        result_file_name = os.path.join(saved_path, str(ID) + "_backward.txt")
        print(ID, len(results), results)
        if len(results) > 0:
            f = open(result_file_name, 'w')
            for pano in results:
                panoId = pano[0]
                pano_lon = pano[1]
                pano_lat = pano[2]
                line = ','.join([str(i) for i in pano])
                f.writelines(line + "\n")
            f.close()
        # print(result['Time_machine'])

def test_saveDepthMap_frm_JsonFile():
    saved_path =  r'L:\Datasets\HamptonRoads\go_along\depthmaps'
    data_path = r'L:\Datasets\HamptonRoads\go_along'
    TXTs = glob.glob(os.path.join(data_path, "*.txt"))
    Jsons = glob.glob(os.path.join(data_path, "*.json"))




    mp_list = mp.Manager().list()
    for j in Jsons:
        mp_list.append(j)

    process_cnt = 8
    pool = mp.Pool(processes=process_cnt)
    for i in range(process_cnt):
        pool.apply_async(save_depthmap, args=(mp_list, saved_path))
    pool.close()
    pool.join()

    # for idx, json in enumerate(Jsons):
    #     try:
    #
    #         gsv.saveDepthMap_frm_JsonFile(json, saved_path)
    #         basename = os.path.basename(json)
    #         print(idx, basename)
    #     except Exception as e:
    #         print("Error in test_saveDepthMap_frm_JsonFile():", e)
    #         continue

def save_depthmap(json_list, saved_path):
    gsv = GPano.GSV_depthmap()
    total = len(json_list)
    processed_cnt = total - len(json_list)
    while len(json_list) > 0:
        json = json_list.pop(0)
        basename = os.path.basename(json)
        basename = basename.replace(".json", '.tif')
        new_name = os.path.join(saved_path, basename)
        if os.path.exists(new_name):
            continue

        gsv.saveDepthMap_frm_JsonFile(json, saved_path)
        processed_cnt += 1
        print(f"Processed {processed_cnt} / {total}")

def read_depthmaps():
    depthmap_path =  r'L:\Datasets\HamptonRoads\go_along\depthmaps'
    data_path = r'L:\Datasets\HamptonRoads\go_along'
    tag = r'_backward'
    back_txts = glob.glob(os.path.join(data_path, "*_backward.txt"))
    saved_path = r'L:\Datasets\HamptonRoads\go_along\heights'
    for idx, txt in enumerate(back_txts[:]):
        print(txt)
        # f = open(txt, 'r')
        df = pd.read_csv(txt, header=None)
        df.columns = ["panoId", 'lon', 'lat']
        # print(df)
        df = df.iloc[::-1]
        df = df[:-1]
        # print(df)

        forward_txt = txt.replace("_backward", '')
        basename = os.path.basename(forward_txt)

        if os.path.exists(forward_txt):
            df_forward = pd.read_csv(forward_txt, header=None)
        df_forward.columns = ["panoId", 'lon', 'lat']
        # print(df_forward)

        df_all = pd.concat([df, df_forward])
        df_all = df_all.reset_index()
        # print(df_all)
        df_all['height'] = -1.0
        df_all['image_date'] = ''
        for idx, row in df_all.iterrows():
            try:
                panoId = row['panoId']
                dm_file = os.path.join(depthmap_path, panoId+".tif")
                json_file = os.path.join(data_path, panoId+".json")
                dm_img = Image.open(dm_file)
                dm_img = np.array(dm_img)
                height = dm_img[-1][255]

                image_date = json.load(open(json_file, 'r'))['Data']['image_date']
                df_all.at[idx, 'height'] = height
                df_all.at[idx, 'image_date'] = image_date
                print(basename, panoId, image_date, dm_img.shape, height)
            except Exception as e:
                print("Error in  df_all.iterrows():", e)
                continue

        # print(df_all)

        df_all = df_all.drop("index", axis=1)
        df_all.to_csv(os.path.join(saved_path, basename), index=False)



    # forward_txts =

def getHeightVariance():
    data_dir = r'L:\Datasets\HamptonRoads\go_along\heights'
    txts = glob.glob(os.path.join(data_dir, '*.txt'))
    txts = natsorted(txts)
    # df_std = pd.DataFrame(columns=['ID', 'start', 'end', 'mean', 'std'])
    row_list = []
    for idx0, txt in enumerate(txts[:]):
        print("Processing:", txt)
        basename = os.path.basename(txt)
        df = pd.read_csv(txt)
        pre_date = '[0000, 00]'
        end_pos = 0
        # start_pos = 0

        sum_height = 0
        count = 0
        std_count = 0


        start_pos = 0
        for idx, row in df[:].iterrows():

            height = row['height']
            sum_height += height
            count += 1
            crt_date = row['image_date']

            crt_pos = idx

            if idx ==0:
                pre_date = crt_date

            if (crt_date != pre_date) or (idx == len(df)-1):

                if crt_pos > end_pos + 1:
                    end_pos = idx - 1
                    std_dict = {}
                    std_dict['ID'] = basename
                    std_dict['image_date'] =crt_date
                    std_dict['start'] = start_pos
                    std_dict['end'] = end_pos
                    std_dict['count'] = end_pos - start_pos +1
                    if std_dict['count']  > 1:
                        std_dict['std'] = df[start_pos:end_pos-1]['height'].std()
                        std_dict['mean']= df[start_pos:end_pos-1]['height'].mean()
                    else:
                        std_dict['std'] = 0
                        std_dict['mean'] = 0
                    start_pos = idx
                    pre_date = crt_date
                    row_list.append(std_dict)
                else:
                    start_pos = idx + 1

    df_std = pd.DataFrame(row_list)
    print(df_std)
    df_std.to_csv(os.path.join(data_dir, "height_std.csv"), index=False)


def getPointCloud_from_PanoId(panoId="", lat=None, lon=None, color=True, saved_path=''):
    # lat, lon = 40.7303117, -74.1815408
    saved_path = r'K:\Research\street_view_depthmap'
    # panoId = r"BM1Qt23drK3-yMWxYfOfVg"
    if (lat != None) and (lon != None):
        panoId, _, _ = GPano.GPano().getPanoIDfrmLonlat(lon, lat)



    dm, jdata = gsv.saveDepthMap_frm_panoId(panoId=panoId, saved_path=saved_path)

    image_date = jdata['Data']['image_date']
    print(panoId, image_date)

    tilt_yaw_deg = jdata['Projection']['tilt_yaw_deg']
    pano_yaw_deg = jdata['Projection']['pano_yaw_deg']
    tilt_pitch_deg = jdata['Projection']['tilt_pitch_deg']
    elevation_egm96_m = jdata['Location']['elevation_egm96_m']

    jpg = os.path.join(saved_path, panoId + '.jpg')
    img = Image.open(jpg)

    dm_np = np.array(dm).astype(float)
    img_np = np.array(img)[0:256, :, :]
    colors = img_np.reshape(-1, 3)
    # print(colors.shape)
    # print(colors)
    # print(img_np.shape)
    nx, ny = (512, 256)
    x_space = np.linspace(0, nx - 1, nx)
    y_space = np.linspace(ny - 1, 0, ny)

    xv, yv = np.meshgrid(x_space, y_space)

    thetas = yv / ny * np.pi - np.pi / 2
    phis = xv / nx * (np.pi * 2) - np.pi
    thetas_sin = np.sin(thetas)
    thetas_cos = np.cos(thetas)
    phis_sin = np.sin(phis)
    phis_cos = np.cos(phis)
    R = thetas_cos * dm_np

    z = dm_np * thetas_sin
    x = R * phis_cos
    y = R * phis_sin

    P = np.concatenate([y.ravel().reshape(-1, 1), x.ravel().reshape(-1, 1), z.ravel().reshape(-1, 1)], axis=1)

    rotate_x_radian = (90 - tilt_yaw_deg) / 180 * math.pi
    rotate_z_radian = (90 - pano_yaw_deg) / 180 * math.pi
    rotate_y_radian = (tilt_pitch_deg) / 180 * math.pi

    P = P.dot(gsv.rotate_x(rotate_x_radian))
    P = P.dot(gsv.rotate_y(rotate_y_radian))
    P = P.dot(gsv.rotate_z(rotate_z_radian))



    if color:
        P = np.concatenate([P, colors], axis=1)
    # print(P.shape, P)

    return P

def showNeighbor_pointClouds(panoId="", lat=40.7303117, lon=-74.1815408, color=True, saved_path=''):
    lat, lon = 40.7301114, -74.1806459
    lat, lon =40.7083166,-74.2559286
    lat, lon = 40.6677323,-74.1797492
    lat, lon = 40.6580514,-74.2094375   # large error
    lat, lon = 40.6594416,-74.2086
    lat, lon = 40.679933,-74.277889
    lat, lon = 40.6749695,-74.2957822
    lat, lon = 40.6740651,-74.2958849
    lat, lon = 40.7068861,-74.2569793
    lat, lon = 40.7064717,-74.2576984
    lat, lon = 40.7092409,-74.2527648
    lat, lon =40.7092581,-74.2452077
    lat, lon =40.672149,-74.1862878
    lat, lon =40.6650419,-74.1821079  # mosaic precision is very bad on bridge.
    lat, lon = 40.6645449,-74.1825446
    lat, lon = 40.6511743,-74.2034643

    jdata = gpano.getPanoJsonfrmLonat(lon, lat)

    panoId = jdata['Location']['panoId']


    P = getPointCloud_from_PanoId(panoId = panoId)

    elevation_egm96_m = jdata['Location']['elevation_egm96_m']
    elevation_egm96_m = float(elevation_egm96_m)
    P[:, 2:3] += elevation_egm96_m


    # crs_4326 = CRS.from_epsg(4326)
    crs_local =  CRS.from_proj4(f"+proj=tmerc +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    print(crs_local)

    transformer = Transformer.from_crs(4326, crs_local)
    new_center = transformer.transform(lat, lon)
    print("new_center: ", new_center)

    links = jdata.get("Links", [])
    for link in links:
        p = link["panoId"]
        pts = getPointCloud_from_PanoId(panoId = p)

        jdata1 = gpano.getJsonfrmPanoID(p)
        lat1 = jdata1['Location']['lat']
        lon1 = jdata1['Location']['lng']
        elevation_egm96_m = jdata1['Location']['elevation_egm96_m']
        elevation_egm96_m = float(elevation_egm96_m)

        new_center = transformer.transform(lat1, lon1)
        pts[:, 0:1] += new_center[1]
        pts[:, 1:2] += new_center[0]
        pts[:, 2:3] += elevation_egm96_m
        print("new_center: ", new_center)

        P = np.concatenate([P, pts], axis=0)


    # print(P[:, :3])
    # print(P[:, 3:6])
    print(P.shape)

    v = pptk.viewer(P[:, :3])
    v.attributes(P[:, 3:6] / 255.)
    v.set(point_size=0.03, show_axis=True, show_grid=True)


    pass


def showPointCloud():

    lon =-74.1628251
    lat =40.7282837
    lat, lon = 40.7316559, -74.1623911
    lat, lon = 40.758635,-73.9637293
    lat, lon = 40.7605202,-73.9640951
    lat, lon = 40.7692807,-73.9435982
    lat, lon = 40.780667,-73.9610365
    lat, lon = 40.7303117,-74.1815408
    saved_path = r'K:\Research\street_view_depthmap'
    # panoId = r'1_LEWPWwTwKBqDXQODfRmA'
    panoId, _, _ = GPano.GPano().getPanoIDfrmLonlat(lon, lat)
    print(panoId)

    # jpg = tif.replace("noflip.tif", '.jpg')
    dm, jdata = gsv.saveDepthMap_frm_panoId(panoId=panoId, saved_path=saved_path)

    tilt_yaw_deg = jdata['Projection']['tilt_yaw_deg']

    # tif = r'I:\t\depthmap0\6kPwaDvg4AZ39ESx1FK13gnoflip.tif'

    # dm = Image.open(tif)
    jpg = os.path.join(saved_path, panoId+'.jpg')
    img = Image.open(jpg)

    dm_np = np.array(dm)
    img_np = np.array(img)[0:256, :, :]
    colors = img_np.reshape(-1, 3)
    print(colors.shape)
    print(colors)
    # print(img_np.shape)
    nx, ny = (512, 256)
    x_space = np.linspace(0, nx-1, nx)
    y_space = np.linspace(ny-1,0, ny)

    xv, yv = np.meshgrid(x_space, y_space)

    # sin = (np.ones(ny, nx))

    thetas = yv / ny * np.pi - np.pi / 2
    phis = xv / nx * (np.pi * 2) - np.pi
    thetas_sin = np.sin(thetas)
    thetas_cos = np.cos(thetas)
    phis_sin = np.sin(phis)
    phis_cos = np.cos(phis)
    R = thetas_cos * dm_np

    z = dm_np * thetas_sin
    x = R * phis_cos
    y = R * phis_sin
    # print(x)
    # print(y)
    # print(z.shape)
    # print(z.ravel().shape)
    # print(z.ravel().reshape(-1, 1).shape)

    #
    # print(xv)
    # print(yv)
    # print(xv.shape)
    # print(yv.shape)



    P = np.concatenate([y.ravel().reshape(-1, 1), x.ravel().reshape(-1, 1), z.ravel().reshape(-1, 1)], axis=1)

    rotate_x_radian = (90 - tilt_yaw_deg) / 180 *math.pi

    P = P.dot(gsv.rotate_x(rotate_x_radian))

    v = pptk.viewer(P)
    v.attributes(colors/255.)
    v.set(point_size=0.03,show_axis=True, show_grid=True)


if __name__ == '__main__':
    # showPointCloud()
    # getPointCloud_from_PanoId()
    # getHeightVariance()
    showNeighbor_pointClouds()
    # read_depthmaps()
    # test_go_along_road_forward()


    # getJsonDepthmapfrmLonlat(-74.317149454999935, 40.798423060000061, dm=1,
    #                          saved_path=r'J:\Sidewalk')

    # saved_path = r'D:\Code\StreetView\Essex\json'
    # #saved_path = r'J:\Sidewalk\t'
    # df = pd.read_csv(r'D:\Code\StreetView\Essex\Essex_10m_points.csv')
    # df = df.fillna(0)
    # df = df[170000:]
    # print('len of df:', len(df))
    # mp_list = mp.Manager().list()
    # for idx, row in df.iterrows():
    #     mp_list.append([row.lon, row.lat, str(int(row.id)), str(int(idx))])
    # print('len of mp_list:', len(mp_list))
    # print('Started!')
    # try:
    #     getJsonDepthmapsfrmLonlats_mp(mp_list, dm=1, saved_path=saved_path, Process_cnt=10)
    #
    # except Exception as e:
    #     print("Error in row: ",  e)


