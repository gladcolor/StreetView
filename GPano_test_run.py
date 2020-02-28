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

    # csv file needs POINT_X,POINT_Y.
    pts = gpano.readRoadSeedsPts_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\House\maryland\Maryland_Road_Centerlines__Comprehensive\Maryland_Road_Centerlines_pts.csv')
    # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
    #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
    coords = gpano.readCoords_csv(r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\House\maryland\maryland_bou.csv')
    polygon = gpano.formPolygon(coords)
    saved_path = r'J:\Maryland\jsons'
    random.shuffle(pts)

    # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
    gpano.getPanoJPGfrmArea_mp('json_only', pts, saved_path, coords, zoom=4, Process_cnt=8)


# def get_trees():


if __name__ == '__main__':
    test_getPanoJPGfrmArea()