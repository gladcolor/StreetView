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


if __name__ == '__main__':
    try:
        test_getPanoJPGfrmArea()
    except:
        test_getPanoJPGfrmArea()