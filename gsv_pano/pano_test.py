import unittest
import math
import os
import numpy as np

import pptk

from pano import GSV_pano

# gsv = GSV_pano()

class TestPano(unittest.TestCase):

    # OK, 2020-12-28
    # def test_init(self):
    #     panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"  # NJIT kinney street
    #     pano1 = GSV_pano(panoId=panoId_2019)
    #     print(pano1)
    #     self.assertEqual(pano1.panoId, panoId_2019)

    # OK, 2020-12-28
    # def test_getPanoIDfrmLonlat(self):
    #     lat, lon = 40.7303117, -74.1815408  # NJIT kinney street
    #     gsv = GSV_pano()
    #     panoId = gsv.getPanoIDfrmLonlat(lon, lat)[0]
    #     panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"
    #     self.assertEqual(panoId, panoId_2019)


    # def test_get_depthmap(self):
    #     panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"
    #     pano1 = GSV_pano(panoId=panoId_2019)
    #     dm = pano1.get_depthmap()
    #     mid_column_sum = dm[:, 127].sum()
    #     print(mid_column_sum)
    #     self.assertEqual(round(mid_column_sum, 5), 864.18066)

    # OK, 2020 - 12 - 29
    # def test_get_point_cloud(self):
    #     panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"
    #     pano1 = GSV_pano(panoId=panoId_2019)
    #     dm = pano1.get_depthmap()
    #     point_cloud = pano1.get_point_cloud(distance_threshole=20, zoom=3)['point_cloud']
    #     P = point_cloud
    #     v = pptk.viewer(P[:, :3])
    #     v.attributes(P[:, 4:7] / 255.0, P[:, 3], P[:, 8:11]/255.0, P[:, 7])
    #     # color_map = np.random.rand(255, 3)
    #     # v.color_map(color_map)
    #     # P = np.concatenate([P, colors, planes, normalvectors], axis=1)
    #     mid_column_sum = dm[:, 127].sum()
    #     v.set(point_size=0.001, show_axis=True, show_grid=True)
    #     self.assertEqual(round(mid_column_sum, 5), 864.18066)

    # OK, 2020-12-28
    # def test_get_panorama(self):
    #     panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"
    #     saved_path = os.getcwd()
    #     pano1 = GSV_pano(panoId=panoId_2019, saved_path=saved_path)
    #
    #     # pano_zoom1 = pano1.get_panorama(zoom=5)
    #     # print("image size:", pano_zoom1.size)
    #     # self.assertEqual(pano_zoom1.size, (16384, 8192))
    #     pano_zoom1 = pano1.get_panorama(zoom=0)
    #     print("image size:", pano_zoom1.size)
    #     self.assertEqual(pano_zoom1.size, (512, 256))
    #     pano_zoom1 = pano1.get_panorama(zoom=4)
    #     print("image size:", pano_zoom1.size)
    #     self.assertEqual(pano_zoom1.size, (16384/2, 8192/2))
    #
    #     pano_zoom1 = pano1.get_panorama(zoom=3)
    #     print("image size:", pano_zoom1.size)
    #     self.assertEqual(pano_zoom1.size, (16384/4, 8192/4))


    # Not OK, 2020 - 12 - 29
    def test_get_DEM(self):
        panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"

        # lat, lon = 40.7092976, -74.2531686  # Millrun Manor Dr.
        # lat, lon = 33.9951421,-81.0254529 # Bull St. Callcot, UofSC
        lat, lon = 33.9977081,-81.0236725 # Henderson St. UofSC
        lat, lon = 33.9901799,-81.0181874 # Enoree Ave. UofSC
        lat, lon = 33.9888126,-81.0156712 # South Greg. UofSC
        lat, lon = 33.9889036,-81.0157056 # South Greg. UofSC
        lat, lon = 40.7122216,-74.2551131 # 1971 Ostwood Terrace, Millrun, Union, NJ
        lat, lon = 40.712275,-74.2552067 # 1971 Ostwood Terrace, Millrun, Union, NJ
        lat, lon = 40.7123314,-74.2553002 # 1971 Ostwood Terrace, Millrun, Union, NJ

        lat, lon = 40.7065092, -74.2565972  # Near Franklin elem. school, NJ
        # lon, lat = -77.0685390, 38.9265898  # Watchington, DC.
        lat, lon = 40.7068861, -74.2569793  # to Franklin elem.

        pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=os.getcwd())
        # pano1 = GSV_pano(panoId=panoId_2019, saved_path=os.getcwd())
        # dm = pano1.get_depthmap()
        # DEM = pano1.get_DEM(width=40, height=40, resolution=0.4, zoom=4)
        # self.assertEqual(DEM.shape, (1333, 1333))

        point_cloud = pano1.get_DOM(width=40, height=40, resolution=0.03, zoom=4)['DOM']
        P = point_cloud
        v = pptk.viewer(P[:, :3])
        v.set(point_size=0.01, show_axis=True, show_grid=False)

        v.attributes(P[:, 3:6] / 255.0)
        # v.attributes(P[:, 4:7] / 255.0, P[:, 3], P[:, 8:11]/255.0, P[:, 7])
        # color_map = np.random.rand(255, 3)
        # v.color_map(color_map)
        # P = np.concatenate([P, colors, planes, normalvectors], axis=1)
        # mid_column_sum = dm[:, 127].sum()



    # def test_get_DOM(self):
    #     panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"  # NJIT kinney street
    #     # panoId_2019 = "LF85GNgr34Rs4wy0_a6lkQ"  # NJIT kinney street
    #     # panoId_2019 = "A3ABgCfEs9T5_TNGkFteXw"  # NJIT kinney street
    #     # lat, lon = 40.7065092, -74.2565972  # Near Franklin elem. school, NJ
    #     # lat, lon = 40.7303117, -74.1815408  # NJIT kinney street
    #
    #     # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=os.getcwd())
    #     pano1 = GSV_pano(panoId=panoId_2019, saved_path=os.getcwd())
    #
    #     DOM = pano1.get_DOM(width=30, height=30, resolution=0.05, zoom=0)
    #
    #     self.assertEqual(DOM.shape, (600, 600))