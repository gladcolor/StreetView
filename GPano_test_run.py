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
    pts = gpano.readRoadSeedsPts_csv(r'G:\Shared drives\sidewalk\Street_trees\Philly\Philly_road_pts.csv')
    # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
    #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
    coords = gpano.readCoords_csv(r'G:\Shared drives\sidewalk\Street_trees\Philly\Philly__boundary.csv')
    polygon = gpano.formPolygon(coords)
    saved_path = r'D:\Research\tree\datasets\Philly\jsons'
    random.shuffle(pts)

    # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
    gpano.getPanoJPGfrmArea_mp('json_only', pts, saved_path, coords, zoom=4, Process_cnt=5)


if __name__ == '__main__':
    test_getPanoJPGfrmArea()