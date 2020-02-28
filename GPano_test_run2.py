from GPano import *
import GPano
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import  sqlite3

gpano = GPano.GPano()
gsv = GPano.GSV_depthmap()


def test_getPanoJPGfrmArea():
    print('started! ')
    pts = gpano.readRoadSeedsPts_csv(r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_county\road_seeds.csv')
    # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
    #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
    coords = gpano.readCoords_csv(r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_county\essex_vet.csv')
    polygon = gpano.formPolygon(coords)
    saved_path = r'G:\My Drive\Sidewalk_extraction\Essex\jsons'
    random.shuffle(pts)

    # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
    gpano.getPanoJPGfrmArea_mp('json_only', pts,  saved_path, coords, Process_cnt=1)


    #### Test for getPanoIDfrmLonlat()
def test_getPanosfrmLonlats_mp():


    list_lonlat = pd.read_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\House\maryland\footprints\footprints.csv')
    print(sys.getfilesystemencoding())
    print(sys.getdefaultencoding())

    mp_lonlat = mp.Manager().list()
    ns = mp.Manager().Namespace()
    ns.list_lonlat = list_lonlat
    print(len(list_lonlat))
    for idx, row in list_lonlat[:10].iterrows():
        mp_lonlat.append([row['POINT_X'], row['POINT_Y'], int(idx + 1)])

    print(len(mp_lonlat))
    gpano.shootLonlats_mp(mp_lonlat, saved_path=r'J:\Maryland\MS_building\images', Process_cnt=2)


if __name__ == '__main__':
    try:
        # test_getPanoJPGfrmArea()
        test_getPanosfrmLonlats_mp()

    except:
        # test_getPanoJPGfrmArea()
        test_getPanosfrmLonlats_mp()