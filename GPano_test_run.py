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
    pts = gpano.readRoadSeedsPts_csv(r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\NewYorkCity\NYC Street Centerline\NYC_road_pts.csv')
    # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
    #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
    coords = gpano.readCoords_csv(r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\NewYorkCity\Borough_Boundaries\NYC__boundary.csv')
    polygon = gpano.formPolygon(coords)
    saved_path = r'G:\My Drive\Sidewalk_extraction\NewYorkCity\Panos_z4'
    random.shuffle(pts)

    # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
    gpano.getPanoJPGfrmArea_mp(None, pts, saved_path, coords, zoom=4, Process_cnt=8)


if __name__ == '__main__':
    test_getPanoJPGfrmArea()