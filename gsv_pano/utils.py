import time

import requests
import json
import base64
import struct
import PIL
import glob
import urllib
from PIL import Image
import os
# import fiona
import multiprocessing as mp
import random

from pyproj import Transformer
from numpy import linalg as LA
import datetime
import pandas as pd
import numpy as np
import zlib
import math
from tqdm import tqdm
import geopandas as gpd

import logging
logging.basicConfig(filename="info.log", level=logging.INFO, format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s")


logging.shutdown()

def refactorJson(jdata):
    newJson = {}

    if len(str(jdata)) < 1000:
        return ""

    try:

        # restore the first level keys
        newJson['Data'] = {}
        newJson['Projection'] = {}
        newJson['Location'] = {}
        newJson['Links'] = {}
        newJson['Time_machine'] = {}
        newJson['model'] = {}


        # Data
        try:
            newJson['Data']['image_width'] = jdata[1][0][2][2][1]
            newJson['Data']['image_height'] = jdata[1][0][2][2][0]
            newJson['Data']['tile_width'] = jdata[1][0][2][3][1][0]
            newJson['Data']['tile_height'] = jdata[1][0][2][3][1][1]
            newJson['Data']['level_sizes'] = jdata[1][0][2][3][0]
            newJson['Data']['image_date'] = jdata[1][0][6][7]
            newJson['Data']['imagery_type'] =  jdata[1][0][0][0]
            newJson['Data']['copyright'] =  jdata[1][0][4][0][0][0][0]
        except Exception as e:
            print("Error in obtain new Json['Data']:", e)

        # Location
        try:
            newJson['Location']['panoId'] = jdata[1][0][1][1]
            newJson['Location']['zoomLevels'] = ''
            newJson['Location']['lat'] = jdata[1][0][5][0][1][0][2]
            newJson['Location']['lng'] = jdata[1][0][5][0][1][0][3]
            newJson['Location']['original_lat'] = ''
            newJson['Location']['original_lng'] = ''
            newJson['Location']['elevation_wgs84_m'] = ""
        except Exception as e:
            # print("Error in obtain new Json['Location']:", e)
            print("Error in obtain new Json['Location']:", e, newJson['Location']['panoId'])
        try:

            newJson['Location']['streetRange'] = ''
            newJson['Location']['country'] = ''
            newJson['Location']['region'] = ''
            newJson['Location']['elevation_egm96_m'] = jdata[1][0][5][0][1][1][0]

        except Exception as e:
            # print("Error in obtain new Json['Location']:", e)
            print("Error in obtain new Json['Location'] 2:", e, newJson['Location']['panoId'])


        # Projection
        try:
            newJson['Projection']['projection_type'] = 'spherical'
            newJson['Projection']['pano_yaw_deg'] =    float(jdata[1][0][5][0][1][2][0])  # z axis, yaw
            newJson['Projection']['tilt_yaw_deg'] =    float(jdata[1][0][5][0][1][2][1])   # y-axis, pitch
            newJson['Projection']['tilt_pitch_deg'] =  float(jdata[1][0][5][0][1][2][2])    # x-axis, roll
        except Exception as e:
            print("Error in obtain newJson['Projection']: newJson['Location']", newJson['Location'])

            print("Error in obtain new Json['Projection']:", e, newJson['Location']['panoId'])
            print("Error in obtain new Json['Projection']:", e, jdata[1][0][5][0])



        # Links
        newJson['Links'] = getLinks(jdata)

        # Time_machine
        newJson['Time_machine'] = getTimeMachine(jdata)

        # model
        newJson['model']['depth_map'] = jdata[1][0][5][0][5][1][2]

        newJson['Location']['country'] = ""
        newJson['Location']['description'] = ""
        newJson['Location']['region'] = ""

        try:
            newJson['Location']['country'] = jdata[1][0][5][0][1][4]
        except Exception as e:
            # print("Error in obtain new Json['Location']['country']:", e)
            pass
            # logging.exception("Error in obtain new Json['Location']['country']: %s" % e)


        try:
            newJson['Location']['description'] = jdata[1][0][3][2][0][0]
        except Exception as e:
            # print("Error in obtain new Json['Location']['description']:", e)
            pass
            # logging.exception("Error in obtain new Json['Location']['description']: %s" % e)

        try:
            newJson['Location']['region'] =  jdata[1][0][3][2][1][0]
        except Exception as e:
            pass
            # print("Error in obtain new Json['Location']['region']:", e)
            # logging.exception("Error in obtain new Json['Location']['region']: %s" % e)

    except Exception as e:
        print("Error in refactorJson():", e)

    return newJson

def getTimeMachine(jdata):
    timemachine_list = []
    try:
        pano_list = jdata[1][0][5][0][3][0]
        dates = jdata[1][0][5][0][8]

        if dates is None:
            return timemachine_list

        for day in dates:
            old_pano_dict = {}
            idx = day[0]

            raw_pano_info = pano_list[idx]

            old_pano_dict['panoId'] = raw_pano_info[0][1]
            old_pano_dict['image_date'] = day[1]

            old_pano_dict["lng"] = raw_pano_info[2][0][3]
            old_pano_dict['lat'] = raw_pano_info[2][0][2]
            old_pano_dict['elevation_egm96_m'] = raw_pano_info[2][1][0]

            old_pano_dict['heading_deg'] = raw_pano_info[2][2][0]
            old_pano_dict['yaw_deg'] = raw_pano_info[2][2][1]
            old_pano_dict['pitch_deg'] = raw_pano_info[2][2][2]

            old_pano_dict['description'] = ''
            if len(raw_pano_info) == 4:
                old_pano_dict['description'] = raw_pano_info[3][2][0][0]

            timemachine_list.append(old_pano_dict)
            # print(link_dict)
        return timemachine_list

    except Exception as e:
        # logging.exception("Error in getLinks().")
        return timemachine_list




def getLinks(jdata):
    link_list = []
    try:
        pano_list =   jdata[1][0][5][0][3][0]
        jdata_links = jdata[1][0][5][0][6]
        link_list = []
        for link in jdata_links:
            link_dict = {}
            idx = link[0]
            raw_pano_info = pano_list[idx]
            yawDeg = link[1][3]
            panoId = raw_pano_info[0][1]
            # print(panoId)
            # print(link, idx, yawDeg)
            link_dict['panoId'] = panoId
            link_dict['yawDeg'] = yawDeg
            link_dict['road_argb'] = ""
            link_dict['description'] = ""
            if len(raw_pano_info) == 4:
                link_dict['description'] = raw_pano_info[3][2][0][0]

            link_list.append(link_dict)
            # print(link_dict)
        return link_list

    except Exception as e:
        logging.exception("Error in getLinks().")
        logging.exception(jdata_links)
        return link_list

# def sort_pano_links(links_dict):
#     new_dict = {}
#     yaw_in_links = [float(link['yawDeg']) for link in links_dict]
#     pano_yaw_deg = float(pano_yaw_deg)
#     diff = [abs(yawDeg - pano_yaw_deg) for yawDeg in yaw_in_links]
#     idx_min = diff.index(min(diff))
#     forward_link = Links[idx_min]
#
#     # find the backward_link
#     idx_max = diff.index(max(diff))
#     backward_link = Links[idx_max]
#
#     #                 print(idx_max, idx_min)
#
#     for link in [forward_link, backward_link]:
#         yawDeg = str(link.get('yawDeg', '-999'))
#         panoId = link.get('panoId', '0')
#         road_argb = link.get('road_argb', '0x80fdf872')
#         description = link.get('description', '-').replace(",", ';')
#         if description == '':
#             description = '-'
#         contents = [yawDeg, panoId, road_argb, description]
#         contents = ',' + ','.join(contents)
#         list_txt.writelines(contents)
#
#     if len(Links) > 2:
#         if idx_max < idx_min:
#             idx_max, idx_min = idx_min, idx_max
#
#         Links.pop(idx_max)
#         Links.pop(idx_min)
#
#         list_txt.writelines(',')
#         content_list = []
#         for link in Links:
#             yawDeg = str(link.get('yawDeg', '-999'))
#             panoId = link.get('panoId', '0')
#             road_argb = link.get('road_argb', '0x80fdf872')
#             description = link.get('description', '-').replace(",", ';')
#             if description == '':
#                 description = '-'
#             contents = [yawDeg, panoId, road_argb, description]
#             content_list.append('|'.join(contents))

    # return new_dict




    # print("ok")

def compressJson(jdata):

    try:
        del jdata[1][0][5][0][5][3][2]

        # cannot compress it yet. The following code works fine, but cannot store the base64 string in the json.
        # print("len(jdata[1][0][5][0][5][1][2]):", len(jdata[1][0][5][0][5][1][2]))

        string_bytes = jdata[1][0][5][0][5][1][2].encode("ascii")

        # print("len(string_bytes):", len(string_bytes))
        compressed_string = zlib.compress(string_bytes)
        # print("compressed_string:", compressed_string)
        # print("len(compressed_string):", len(compressed_string))
        # print("compressed_string:", compressed_string.decode("utf-8"))

        # compressed_string = b"eJzt131wFPUdx_G7EEJSNJeExzwQCCGCITwkAi3NLrsJtlEL2lY6MC3WYaYwOFiR-gAWyRZaE8WWhwpalKEgtII81Maq0Fz4gaWtBYtt6aApIk-llAFqisgodZjeQx4vd7_bvdvffXdvP-9_mMll5va3r88tl_SJLleKy53uQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQv5SOkd9MUh8KbqjvlJkavrhsYPkKmZ5rMDumUKPEdgyk-2xARslyB4bsEOC8TEBK5cQfEzAmiUQHxOwWgnHxwQsFJE-FmCJCPWxAPKI9f1R3wIHR03fGvVtcGjU7J2ivhUOjJo8JOrb4bCoucNEfUucFLV12KhvimOiho4Y9Y1xRtTKkaO-M06I2piXh_rmJH_UxNzgLzpqYX7wFxw1cJTgLzhq4CjBX3DUwFGCv-CogaPkwQAERy3MD_6ioxbmB3_RUQvzg7_oqIX5wV901ML84C86amF-8BcdtTA_-IuOWpgf_EVHLcwP_qKjFuYHf9FRC_ODv-iohfnBX3TUwvzgLzpqYX7wFx21MD_4i45amB_8RUctzA_-oqMW5gd_0VEL84O_6KiF-cFfdNTC_OAvOmphfvAXHbUwP_iLjlqYH_xFRy3MD_6ioxbmB3_RUQvzg7_oqIX5wV901ML84C86amF-8BcdtTA_-IuOWpgf_EVHLcwP_qKjFuYHf-FRE3ODv_CoibnBX3jUxNzgLzxqYm7wFx41MTf4C4-amBv8hUdNzA3-wqMm5kbu3zNSxNdlXtTE3Gj8I6KHLyuL4iLNipqYW2L9Dbp3-GfZeALUxNwS5B8jPPxFlwD_-OzhLzT4C4-amBv8hUdNzA3-wqMm5maqf_g_2uFv4UzzD0pFfgX-lswU_w4p_qvwt1wm-HeWivY6_C2WeP_4-eEvLvgLj5qYG_yFR03MDf7CoybmZty_b9-QH8CfHzUxN2P-A_v6C_kh_Dnl5eVRE3Pz-_frp-8sAwfC32j28Ne3APgbzy7-egYAf-PBH_5WDv5is7H_YNfgwV1-AH_9DW0N_snpf0O3cnNzC9sqbg3-dvfv7swvN1j7DuBvE__P8YtpBb7ycgupibnp8q8I_qPDf1Brw4b5XsgJROzfK1AUXQMZXwE1MTePp6ws4F86ZMyY0aNdI0eO9N20sWPLXKWlpUPaco0YMaKkpM2_xFd-fhv1oM5SOZ3K9ldQYJZ_747Se8Vewpdgdf_y8oD_cH_jvzBuws2-BnRvWLCb_OV0qYt_dmsFBQU3BhPgH9iArzhmYO4WuAOgJubm8eTkBPxHjRpV1F6OgT6f3VFBB3ug_r7M8k_1FWYDZuzA7DV0_R-ImpibxxPmsz4gP6J2dphC0Nvg_fXolAn-nA2Yt4MYNhHpy0eGP2pibh5PfphuDMccQt7NPCK8gdxhC_hnZqamRthAlxGIGUL3In_lzOgaNTE3jycsY9QPeCT4GN35_r4yg6WmRhxB6AaELUGPuc39s6Nym_R51-0fdgHRHwSmjkGXtu38wz7fE_NpN-QfaQJGR6BjGWF-wZB4l1Kpibnp9xckbsQ_4gLMGUHkYqUPRE3MLTy_YOXY_XkL0P2dIAH6XS6K2piXh4w6bNH9o07A7EdBPPKW34DF-PX565iAaTOIX97SG7Crv74FxD0Dc9wtuwFq7e7p9zcygRiHYCK79UZALR0-Q_7GJxBpCuHGYLK5hUZAjczL6ABinoA1gn1Ixv3bF2DTCfgDfVux-Nv-KdCWo-WDxeqfFBNIS0tzrnywOPztP4G01pwI31p8_jbfQJt_WsdpokJ3_RVqvriL399t3y-EYfz11ScQtZ0ZmeJv1w3E7N-2APtPwDT_ziOgdtVbHP7uJHkImOvvz0YPgvj8k-IhYCp9R_YYQbz-bvtPwDzycFl8Bib4u20-AZOg-Vl1Bub4u-08ATN4jZSZaaExmMUfyJ4TMOXo8ZcE_m5bPgbMOnqcJYe_P5ttwMyjx1Hy-Puz0QbMPnqMJZd_IHuMQMjRjZeE_sGsvgKBRzdS0voHs-4KhB9dX0nu316f8Etwk9gH3tkaOcU_UvCHP_zhD3_4wx_-8Ic__OEPf_g7YQCk_PAnD_7whz_84Q9_-MMf_vCHP_zhD3_4w1-oP_Xh3fCHP_zhD3_4wx_-8Ic__OEPf_jDH_7wh39yDwD-8Kfihz998Ic__OEPf_jDH_6J9Kc-uz_4wx_-8Ic__OEPf_jDH_7wd8oACPnhb4HgD3_4wx_-8E-wP_XRA8Ef_vCHP_zh78QBwB_-NPzwhz998Kfypz55MPjDH_7wd-oA4A9_En74O9uf-uCtwR_-8Ie_UwdAxQ9_Z_tTn7st-MPfyQMg4oe_s_2pj90e_OHv5AHQ8FvHn3wAxMen8Sc-dOeo_YkHQMIPf2f70x65a9T8xANwOr8F_EkHQMAP_9AIj-90fkv4Ew7A6fzW8KcbgNP5LeJPNoAE81MdM3LU8O3RHD-h-hbkt44_zQIcrm8pf4oFJEw_8UfTFzV5aIk9vcPx3dbz95e40zvb3h-1deQScXqh8lanD0bNHDWRhxfiLvKCzY-a12hmnt2R4KFRgwpI58mT31Zf1Fw0Ud91C0ZNkoio77GlcoV0094zQ-veWO594arGNvbNU39y53xl-bbX9u5ZXi-NOPYP9ejFisb3186QMq5prGpPrvrplErpldc2eOd9qLE9xweotZM1tuvkuqbTFcXeXtK6qju8K0oeeuktueFALXv7WK56R-EJaf3H871bT2qMPZqlzi1skWqOrJBONmts9ZnRqjZtiVT_wRFp0mmNHWy-RS1OuSr9bewU7zePa-y7d2arE-9ZKOWtrm96a67GNs_8WKm4f4B89PZ8b_VfNNZQWqRqFR819pZ-L1Vc0NjPvpKlrvpFhbRx__mmg3fVstSnLyjbry1WTqwdN2n8nlSppe9zSvN5je2v3e91fVruHffmM1V_ri5qvHy2WP7qIY3NPnlNeXHa4crdYx6QD6_U2IbH9inji4c3Zo_Ok6f_VWMrs3PUXy1OZy_9dAa7ujRdOnfzrdW3rNGUNY_Mbtr3Xo00Y2FW1bKMYY3bt5z2vvEvjWUOldXiE3WVK1Pq5Ovf01jNgp8rl5bXsjNr35Nd56d5Hxw4uEo5d0Da_Ot___aTixoremeQOv1Uw6TC5vlKwc7JXtZ8tmr237c0HmtYKm9YrLHdLduVL_3hw8ZFn-2Uv5atsXO3rVLcmSu-eDz9VFPRQ7XshR35Vf1TNldO3fdMZd11jblreqqX352vPPJOw94vX62Tru0ur378k3J5WfoDsnK3xtYU9lAPzhgurUpfIs_VNNa0fo_yn3lD5WfX53mf-pPGjhyU1YE9N0kNg2Y1XrmssXXF_dUXz6rSpDHnvN86pbFZB15Wr12_PHH-iJ5Nfq8Fy1ZXff_2ftLd11fJR2ZoTN60VHn9h-nSZ68837S0WmOXTpZVvVsqyfnbviHXztLYDxZMVDMWHZXydjwh773Ht6ff_VK5b9QuaVrJrsrqFo29Wv8_dc7DV6T3mSrVfKCxQ1P6VT32VJp8W_oi-a7v-DxePqds_bRIenzHKPnYfo31fDBL3fzw29L5Jx-VSv_pe7-S9Kp-m0bK9079sTy9TGNpZ8uUZ8s3SK8-sV9a4rv-w_I2dea2XpVDB_ym8YTvfrUUvK4WXZqpFKhutmrfWumPNYuqt1zoI--avEI-OsH3_h_NVe473EOeU7JWvjJGY98ePUFZv65F8g55Wr51isbenFqnbNma0XTo3oNNNzxfq3iVncrGH329cXj9ZXlZRS27v_mS_Fz9f6WlvefJixZqbM6hi8r_AXMBMr0"
        # print("len(compressed_string):", len(compressed_string))


        base64_string = base64.b64encode(compressed_string)
        # base64_string = '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'
        # base64_string = bytes('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'.encode("utf-8"))
        # print("len(base64_string):", len(base64_string))
        # print("base64_string:", base64_string)

        debase64_string = base64.b64decode(base64_string)
        # print("len(debase64_string):", len(debase64_string))

        uncompressed_string = zlib.decompress(debase64_string)
        # print("len(uncompressed_string):", len(uncompressed_string))
        # print("uncompressed_string:", uncompressed_string)

        # print("base64_string:", base64_string.decode("ascii"))
        jdata[1][0][5][0][5][1][2] = base64_string.decode('ascii')
        # jdata[1][0][5][0][5][1][2] = uncompressed_string.decode('ascii')
        # jdata[1][0][10] =

        # jdata[1][0][11] = base64_string.decode('ascii')

    except Exception as e:
        print("Error in compressJson(), error_info:", e)
    return jdata


def rotate_x(pitch):  # verified, good to use, same as wikipedia
    """:param
    pitch: in radians
    """

    r_x = np.array([[1.0,0.0,0.0],
                    [0.0,math.cos(pitch),-1*math.sin(pitch)],
                    [0.0,math.sin(pitch),math.cos(pitch)]])
    return r_x

def rotate_y(yaw):
    #
    r_y = np.array([[math.cos(yaw),0.0,math.sin(yaw)],
                    [0.0,1.0,0.0],
                    [-1*math.sin(yaw),0.0,math.cos(yaw)]])
    return r_y

def rotate_z(roll):
    #
    r_z = np.array([[math.cos(roll),-1*math.sin(roll),0.0],
                    [math.sin(roll),math.cos(roll),0.0],
                    [0.0,0.0,1.0]])
    return r_z


def parseHeader(depthMap):
    return {
        "headerSize": depthMap[0],
        "numberOfPlanes": getUInt16(depthMap, 1),
        "width": getUInt16(depthMap, 3),
        "height": getUInt16(depthMap, 5),
        "offset": getUInt16(depthMap, 7),
    }

def get_bin(a):
    ba = bin(a)[2:]
    return "0" * (8 - len(ba)) + ba

def getUInt16( arr, ind):
    a = arr[ind]
    b = arr[ind + 1]
    return int(get_bin(b) + get_bin(a), 2)

def getFloat32( arr, ind):
    return bin_to_float("".join(get_bin(i) for i in arr[ind: ind + 4][::-1]))

def bin_to_float( binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

def parsePlanes(header, depthMap):
    indices = []
    planes = []
    n = [0, 0, 0]

    # for i in range(header["width"] * header["height"]):  # original
    #     indices.append(depthMap[header["offset"] + i])
    try:
        # huan
        indices = depthMap[header["offset"]:header["width"] * header["height"] + header["offset"]]

        # do not know how to optimize.
        for i in range(header["numberOfPlanes"]):
            byteOffset = header["offset"] + header["width"] * header["height"] + i * 4 * 4
            n = [0, 0, 0]
            n[0] = getFloat32(depthMap, byteOffset)
            n[1] = getFloat32(depthMap, byteOffset + 4)
            n[2] = getFloat32(depthMap, byteOffset + 8)
            d = getFloat32(depthMap, byteOffset + 12)
            planes.append({"n": n, "d": d})


    except Exception as e:
        logging.exception("Error in utils.parsePlanes(): %s" % e,  exc_info=True)
        planes = []
        indices = []

    return {"planes": planes, "indices": indices}

def parse( b64_string):
    # fix the 'inccorrect padding' error. The length of the string needs to be divisible by 4.
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    # convert the URL safe format to regular format.
    data = b64_string.replace("-", "+").replace("_", "/")

    # origninal
    # data = base64.b64decode(data)  # decode the string
    # data = zlib.decompress(data)  # decompress the data

    # Huan
    data = base64.b64decode(b64_string)
    data = zlib.decompress(data)  # decompress the data
    data = data.decode("ascii")
    data += "=" * ((4 - len(data) % 4) % 4)
    data = data.replace("-", "+").replace("_", "/")
    data = base64.b64decode(data)

    return np.array([d for d in data])

def computeDepthMap(header, indices, planes):

    v = [0, 0, 0]
    w = header["width"]
    h = header["height"]

    depthMap = np.empty(w * h)
    normal_vector_map = np.zeros((h, w, 3), dtype=np.uint8)

    sin_theta = np.empty(h)
    cos_theta = np.empty(h)
    sin_phi = np.empty(w)
    cos_phi = np.empty(w)

    for y in range(h):
        theta = (h - y) / h * np.pi  # original
        # theta = y / h * np.pi  # huan
        sin_theta[y] = np.sin(theta)
        cos_theta[y] = np.cos(theta)

    for x in range(w):
        phi = x / w * 2 * np.pi  # + np.pi / 2
        sin_phi[x] = np.sin(phi)
        cos_phi[x] = np.cos(phi)

    plane_idx_map = np.reshape(indices, (h, w))

    for y in range(h):
        for x in range(w):
            planeIdx = indices[y * w + x]

            # Origninal
            # v[0] = sin_theta[y] * cos_phi[x]
            # v[1] = sin_theta[y] * sin_phi[x]

            # Huan
            v[0] = sin_theta[y] * sin_phi[x]
            v[1] = sin_theta[y] * cos_phi[x]
            v[2] = cos_theta[y]

            if planeIdx > 0:
                plane = planes[planeIdx]
                t = np.abs(plane["d"] / (v[0] * plane["n"][0] + v[1] * plane["n"][1] + v[2] * plane["n"][2]))

                normal_vector_map[y, x, 0] = int(plane["n"][0] * 128 + 128)
                normal_vector_map[y, x, 1] = int(plane["n"][1] * 128 + 128)
                normal_vector_map[y, x, 2] = int(plane["n"][2] * 128 + 128)

                # original
                #     depthMap[y * w + (w - x - 1)] = t
                # else:
                #     depthMap[y * w + (w - x - 1)] = 0

                # huan
                if t < 200:
                    depthMap[y * w + x] = t
                else:
                    depthMap[y * w + x] = 0
            else:
                depthMap[y * w + x] = 0

    depthMap = depthMap.reshape((h, w))

    return {"width": w, "height": h, "depthMap": depthMap, "normal_vector_map": normal_vector_map, 'plane_idx_map': plane_idx_map}

def save_a_list(lst, saved_file):
    with open(saved_file, 'w') as f:
        lst_cleaned = [str(line).replace('\n', '') for line in lst]
        f.writelines("\n".join(lst_cleaned))

def epsg_transform(in_epsg, out_epsg):
    # crs_4326 = CRS.from_epsg(4326)
    # crs_local =  CRS.from_proj4(f"+proj=tmerc +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    # print(crs_local)
    return Transformer.from_crs(in_epsg, out_epsg)

def sort_pano_jsons(json_dir, saved_path=''):
    files = glob.glob(os.path.join(json_dir, "*.json"))[:]
    panoIds = [os.path.basename(f)[:-5] for f in files]
    panoIds = sorted(panoIds, reverse=False)
    total_cnt = len(panoIds)
    sorted_panoIds = []
    cnt = 0
    print("Sorting panorama IDs...")
    start_time = time.perf_counter()
    while len(panoIds) > 0:
        try:
            panoId = panoIds.pop()

            cnt += 1
            if cnt % 1000 == 0:
                end_time = time.perf_counter()
                total_time = end_time - start_time
                efficency = total_time / len(sorted_panoIds)
                time_remain = efficency * len(panoIds)
                print(f"Processed {cnt} / {len(files)}")
                print(
                    f"Time spent (seconds): {time.perf_counter() - start_time:.1f}, time used: {delta_time(total_time)} , time remain: {delta_time(time_remain)}  \n")

            sorted_panoIds.append(panoId)
            json_file = os.path.join(json_dir, panoId + ".json")
            jdata = json.load(open(json_file, 'r'))
            Links = jdata['Links']
            for link in Links:
                link_panoId = link['panoId']
                # if link_panoId in sorted_panoIds:
                #     panoIds.append(link_panoId)
                if link_panoId in panoIds:
                    panoIds.remove(link_panoId)
                    panoIds.append(link_panoId)
                    # print(link_panoId)


        except Exception as e:
            print("Error in sort_pano_jsons:", e)
            continue
    print("Sorting panorama IDs finished.")
    if saved_path != "":
        os.makedirs(saved_path, exist_ok=True)
        with open(os.path.join(saved_path, 'sorted_panoIds.txt'), 'w') as f:
            f.writelines('\n'.join(sorted_panoIds))

    return sorted_panoIds


def degree_difference(angle1, angle2):
    diff = abs(angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def find_forward_bacwark_link(Links, pano_yaw_deg):
    if Links:
        Cnt_links = len(Links)

    # find the forward_link
    yaw_in_links = [float(link['yawDeg']) for link in Links]
    pano_yaw_deg = float(pano_yaw_deg)
    diff = [degree_difference(yawDeg, pano_yaw_deg) for yawDeg in yaw_in_links]


    idx_min = diff.index(min(diff))
    forward_link = Links[idx_min]

    # find the backward_link
    idx_max = diff.index(max(diff))
    backward_link = Links[idx_max]
    return (forward_link, backward_link)

def delta_time(seconds):
    delta1 = datetime.timedelta(seconds=seconds)
    str_delta1 = str(delta1)
    decimal_digi = 0
    point_pos = str_delta1.rfind(".")
    str_delta1 = str_delta1[:point_pos]
    return str_delta1

def dir_jsons_to_list(json_dir, saved_name, sort=True):

    if sort:
        sorted_panoIds = sort_pano_jsons(json_dir, saved_path='')
        jsons_list = [os.path.join(json_dir, p + '.json') for p in sorted_panoIds]
    else:
        jsons_list = glob.glob(os.path.join(json_dir, "*.json"))

    list_txt = open(saved_name, 'w', encoding="utf-8")
    list_txt.writelines('image_width,image_height,tile_width,tile_height,image_date,imagery_type,\
projection_type,pano_yaw_deg,tilt_yaw_deg,tilt_pitch_deg,panoId,zoomLevels,lat,lng,original_lat,original_lng,elevation_wgs84_m,\
description,streetRange,region,country,elevation_egm96_m,\
links_cnt,link_forward_yawDeg,link_f_panoId,link_f_road_argb,link_f_description,\
link_backward_yawDeg,link_b_panoId,link_b_road_argb,link_b_description,link_others')
    for idx, file in tqdm(enumerate(jsons_list)):
        try:
            f = open(file, 'r')
            json_data = json.load(f)

            # default values
            image_width = '0'
            image_height = '0'
            tile_width = '0'
            tile_height = '0'
            image_date = '0000-00'
            imagery_type = '1'

            projection_type = '-1'
            pano_yaw_deg = '-999'
            tilt_yaw_deg = '-999'
            tilt_pitch_deg = '-999'

            panoId = '0'
            zoomLevels = '-1'
            lat = '-999'
            lng = '-999'
            original_lat = '-999'
            original_lng = '-999'
            elevation_wgs84_m = '-9999'
            description = '-'
            streetRange = '-'
            region = '-'
            country = '-'
            elevation_egm96_m = '-9999'

            Data = json_data.get('Data')
            if Data:
                image_width = Data.get('image_width', image_width)
                image_height = Data.get('image_height', image_height)
                tile_width = Data.get('tile_width', tile_width)
                tile_height = Data.get('tile_height', tile_height)
                image_date = Data.get('image_date', image_date)
                image_date = str(image_date).replace(", ", "-").replace("[", "").replace("]", "")
                imagery_type = Data.get('imagery_type', imagery_type)

            Projection = json_data.get('Projection')
            if Projection:
                projection_type = Projection.get('projection_type', projection_type)
                pano_yaw_deg = Projection.get('pano_yaw_deg', pano_yaw_deg)
                tilt_yaw_deg = Projection.get('tilt_yaw_deg', tilt_yaw_deg)
                tilt_pitch_deg = Projection.get('tilt_pitch_deg', tilt_pitch_deg)

            Location = json_data.get('Location')
            if Location:
                panoId = Location.get('panoId', panoId)
                zoomLevels = Location.get('zoomLevels', zoomLevels)
                lat = Location.get('lat', lat)
                lng = Location.get('lng', lng)
                original_lat = Location.get('original_lat', original_lat)
                original_lng = Location.get('original_lng', original_lng)
                elevation_wgs84_m = Location.get('elevation_wgs84_m', elevation_wgs84_m)
                description = Location.get('description', description).replace(",", ';')
                streetRange = Location.get('streetRange', streetRange)
                region = Location.get('region', region).replace(",", ';')
                country = Location.get('country', country).replace(",", ';')
                elevation_egm96_m = Location.get('elevation_egm96_m', elevation_egm96_m)

            Links = json_data.get('Links')

            contents = [image_width, image_height, tile_width, tile_height, image_date, imagery_type, \
                        projection_type, pano_yaw_deg, tilt_yaw_deg, tilt_pitch_deg, \
                        panoId, zoomLevels, lat, lng, original_lat, original_lng, elevation_wgs84_m, description,
                        streetRange, region, country, elevation_egm96_m]

            contents = [str(x) for x in contents]
            #             f.writelines(','.encode('utf-8', 'ignore'))
            list_txt.writelines('\n' + ','.join(contents))

            if Links:
                Cnt_links = len(Links)

                list_txt.writelines(',' + str(Cnt_links))

                # find the forward_link
                yaw_in_links = [float(link['yawDeg']) for link in Links]
                pano_yaw_deg = float(pano_yaw_deg)
                diff = [abs(yawDeg - pano_yaw_deg) for yawDeg in yaw_in_links]
                idx_min = diff.index(min(diff))
                forward_link = Links[idx_min]

                # find the backward_link
                idx_max = diff.index(max(diff))
                backward_link = Links[idx_max]

                #                 print(idx_max, idx_min)

                for link in [forward_link, backward_link]:
                    yawDeg = str(link.get('yawDeg', '-999'))
                    panoId = link.get('panoId', '0')
                    road_argb = link.get('road_argb', '0x80fdf872')
                    description = link.get('description', '-').replace(",", ';')
                    if description == '':
                        description = '-'
                    contents = [yawDeg, panoId, road_argb, description]
                    contents = ',' + ','.join(contents)
                    list_txt.writelines(contents)

                if len(Links) > 2:
                    if idx_max < idx_min:
                        idx_max, idx_min = idx_min, idx_max

                    Links.pop(idx_max)
                    Links.pop(idx_min)

                    list_txt.writelines(',')
                    content_list = []
                    for link in Links:
                        yawDeg = str(link.get('yawDeg', '-999'))
                        panoId = link.get('panoId', '0')
                        road_argb = link.get('road_argb', '0x80fdf872')
                        description = link.get('description', '-').replace(",", ';')
                        if description == '':
                            description = '-'
                        contents = [yawDeg, panoId, road_argb, description]
                        content_list.append('|'.join(contents))

                    list_txt.writelines('|'.join(content_list))

        except Exception as e:
            print("Error in reading json:", file, '-- ', e)
            continue


    list_txt.close()

def get_around_thumnail_from_bearing(lon=0.0, lat=0.0,
                                     panoId='',
                                     bearing_list=[0.0, 90.0, 180.0, 270.0],
                                     saved_path='', prefix='', suffix='',
                                     width=1024, height=768,
                     pitch=0,  fov=90,
                                     overwrite=True):
    ''':argument

    '''
    # w maximum: 1024
    # h maximum: 768
    server_num = random.randint(0, 3)
    lon = round(lon, 7)
    lat = round(lat, 7)
    height = int(height)
    pitch = int(pitch)
    width = int(width)

    # suffix = str(suffix)
    prefix = str(prefix)
    if prefix != "":
        # print('prefix:', prefix)
        prefix = prefix + '_'
    if suffix != "" and (not isinstance(suffix, list)):
        suffix = '_' + suffix

    elif not isinstance(suffix, list):
        suffix = str(suffix)


    for idx, yaw in enumerate(bearing_list):
        if yaw > 360:
            yaw = yaw - 360
        if yaw < 0:
            yaw = yaw + 360

        if len(panoId) == 22:  # panoId is input
            url1 = f"https://geo{server_num}.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&output=thumbnail&thumb=2&w={width}" \
                   f"&h={height}&pitch={pitch}&panoid={panoId}&yaw={yaw}&thumbfov={fov}"
            # print("URL in getImagefrmAngle():", url1)
        else:
            if lat != 0 and lon !=0:
                url1 = f"https://geo{server_num}.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&output=thumbnail&thumb=2&w={width}" \
                       f"&h={height}&pitch={pitch}&ll={lat}%2C{lon}&yaw={yaw}&thumbfov={fov}"
            else:
                print("Error in get_around_thumnail_from_bearing() getting url1: ", "No lat/lon or valid panoId.")
                # return 0, 0, 0
        # print("URL in getImagefrmAngle():", url1)



        try:

            if isinstance(suffix, list):
                if len(suffix) == len(bearing_list):
                    suffix_element = f"_{suffix[idx]}"
                    jpg_name = os.path.join(saved_path, (prefix + str(lat) + '_' + str(lon) + '_' + str(pitch) + '_' +
                                                         str('{:.2f}'.format(yaw)) + suffix_element + '.jpg'))
            else:
                jpg_name = os.path.join(saved_path, (prefix + str(lat) + '_' + str(lon) + '_' + str(pitch) + '_' +
                                                     str('{:.2f}'.format(yaw)) + suffix + '.jpg'))

            if not overwrite:
                if os.path.exists(jpg_name):
                    print("Skip existing file:", jpg_name)
                    continue

            file = urllib.request.urlopen(url1)
            image = Image.open(file)

            # new_name = f"{prefix}{}_{}_{}{}{}"


            if image.getbbox():
                if saved_path != '':
                    image.save(jpg_name)
                else:
                    # print(url1)
                    pass
                # return image, jpg_name, url1

        except Exception as e:
            print("Error in get_around_thumnail_from_bearing() getting url1", e)
            print(url1)
            # return 0, 0, url1


def bearing_angle(x1, y1, x2, y2):
    # Compute differences
    delta_x = x2 - x1
    delta_y = y2 - y1

    # Calculate initial bearing
    theta = math.atan2(delta_x, delta_y)

    # Convert to degrees
    bearing = math.degrees(theta)

    # Normalize to [0, 360)
    bearing = (bearing + 360) % 360

    return bearing


def row_col_to_angle(row, col, width, height, horizontal_fov_rad):
    '''
    Convert pixel row/col to angle (azimuth and altitude)
    :param row:
    :param col:
    :param width:
    :param height:
    :param horizontal_fov_rad:
    :return:
    '''
    # Normalize pixel coordinates
    x = col - width / 2
    y = height / 2 - row

    azimuth_max = horizontal_fov_rad / 2
    R = (width / 2) / math.tan(azimuth_max)  # in pixel

    azimuth_rad = math.atan(x / R)  # = math.atan(x / R)

    hypotenuse = R / math.cos(azimuth_rad)

    altitude_rad = math.atan(y / hypotenuse)

    # print("x, y, R:", x, y, R)
    # print("azimuth_rad, math.cos(azimuth_rad):", azimuth_rad, math.cos(azimuth_rad))

    return azimuth_rad, altitude_rad


def two_points_distance(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return distance

def two_panos_distance(pano1, pano2):  # not finished
    if pano1.x is not None:
        distance = two_points_distance(pano1.x, pano1.y, pano2.x, pano2.y)

    else:
        print("Need to compute pano1.x/y, pano2.x/y first!")
        return None

    return distance
