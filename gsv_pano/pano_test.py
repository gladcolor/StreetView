import unittest
import math
import os
import numpy as np
import time
import glob


import pptk
import cv2
from pano import GSV_pano
from PIL import Image
import PIL

# gsv = GSV_pano()
import matplotlib.pyplot as plt

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


   # OK, 2021 - 03 - 28
#    def test_get_DEM(self):    # note:
#         panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"  # # NJIT kinney street
#         panoId_2019 = "9t4cfX1WMnqGL9Jcv8TiFQ"
#         panoId_2019 = "ARYuiC08k4hlknJQzrhdHQ"
#         panoId_2019 = "K5hylqKRbrEUUWUnXpEUFQ"
#         panoId_2019 = "6_N2PE5LuVclj7agvuywWw"
#         panoId_2019 = "RFoFa5edI4V_bErU2XCWGQ"
#
#         # lat, lon = 40.7092976, -74.2531686  # Millrun Manor Dr.
#         # lat, lon = 33.9951421,-81.0254529 # Bull St. Callcot, UofSC
#         lat, lon = 33.9977081,-81.0236725 # Henderson St. UofSC
#         lat, lon = 33.9901799,-81.0181874 # Enoree Ave. UofSC
#         lat, lon = 33.9888126,-81.0156712 # South Greg. UofSC
#         lat, lon = 33.9889036,-81.0157056 # South Greg. UofSC
#         lat, lon = 40.7122216,-74.2551131 # 1971 Ostwood Terrace, Millrun, Union, NJ
#         lat, lon = 40.712275,-74.2552067 # 1971 Ostwood Terrace, Millrun, Union, NJ
#         lat, lon = 40.7123314,-74.2553002 # 1971 Ostwood Terrace, Millrun, Union, NJ
#
#         lat, lon = 40.7065092, -74.2565972  # Near Franklin elem. school, NJ
#         lon, lat = -77.0685390, 38.9265898  # Watchington, DC.
#         lat, lon = 40.7068861, -74.2569793  # to Franklin elem.
#
#         zoom = 4
#         json_file = r'D:\Code\StreetView\gsv_pano\v-jZjDLJbQBv5LpKqgIXAA.json'
#
#         pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=os.getcwd(), crs_local=6526)
#         # pano1 = GSV_pano(json_file=json_file, saved_path=os.getcwd(), crs_local=6526)
#         # pano1 = GSV_pano(panoId=panoId_2019, saved_path=os.getcwd(), crs_local=6526)
#
#         # dm = pano1.get_depthmap()
#         # P = pano1.get_DEM(width=40, height=40, resolution=0.03, zoom=zoom)["DEM"]
#         # self.assertEqual(DEM.shape, (1333, 1333))
#
#         # point_cloud = pano1.get_DOM(width=40, height=40, resolution=0.03, zoom=zoom)['DOM']
#         # point_cloud = pano1.get_point_cloud(zoom=zoom)['point_cloud']
#
#
#         # ground_points = pano1.get_ground_points(zoom=zoom)  # looks okay  2021-03-26
#         DOM_resolution= 0.05
#         # DEM = pano1.get_DEM(width=40, height=40, resolution=0.1, dem_coarse_resolution=0.4, zoom=1, smooth_sigma=2)
#         timer_start = time.perf_counter()
# #     thetas, phis = pano1.col_row_to_angles(xv, yv)
# #
#         DOM = pano1.get_DOM(width = 30, height = 30, resolution=DOM_resolution, zoom=zoom, img_type="DOM")
#         # points = pano1.get_DEM(width = 40, height = 40, resolution=DOM_resolution, zoom=zoom)['DEM']
#         timer_end = time.perf_counter()
#         print("Time spent (second):", timer_end - timer_start)
        # points = DOM['DOM_points']
        # P = np.argwhere(points > -100)

        # z = points[points > -1000].ravel().reshape(-1, 1)
        # P = np.concatenate([P, z], axis=1)
        # thetas, phis = pano1.XYZ_to_spherical(P[:, :3])
        # pixels = pano1.find_pixel_to_thetaphi(thetas, phis, zoom=zoom)
        # v = pptk.viewer(points[:, :3])
        # v.set(point_size=0.01, show_axis=True, show_grid=False)
        # v.attributes(pixels /255.0)
        # v.attributes(points[:, 3:6]/255.0 )





    def test_get_DOM(self):
        panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"  # NJIT kinney street
        panoId_2019 = "-ft2bZI1Ial4C6N_iwmmvw"

        # panoId_2019 = "-0D29S37SnmRq9Dju9hkqQ"  # NJIT kinney street

        # panoId_2019 = "LF85GNgr34Rs4wy0_a6lkQ"  # NJIT kinney street
        # panoId_2019 = "A3ABgCfEs9T5_TNGkFteXw"  # NJIT kinney street
        # lat, lon = 40.7065092, -74.2565972  # Near Franklin elem. school, NJ
        # lat, lon = 40.7303117, -74.1815408  # NJIT kinney street
        lat, lon = 40.7084995,-74.2556749  # 1.  Walker Ave to Franklin elem. school, NJ
        lat, lon = 40.7084382,-74.2557599  # 2.  Walker Ave to Franklin elem. school, NJ
        lat, lon = 40.7086017,-74.2555401  # 3.  Walker Ave to Franklin elem. school, NJ

        # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=os.getcwd())

        start_time = time.perf_counter()

        pano1 = GSV_pano(panoId=panoId_2019, crs_local=6526, saved_path=os.getcwd())
        zoom = 4
        DOM_resolution = 0.05



        DOM_points = pano1.get_DOM_points(width=40, height=40, resolution=DOM_resolution, zoom=zoom, img_type="DOM")
        print("Time spent (seconds): ", time.perf_counter() - start_time)
        points = DOM_points
        # print(DOM_points)
        v = pptk.viewer(points[:, :3])
        v.set(point_size=0.01, show_axis=True, show_grid=False)

        v.attributes(points[:, 3:6]/255.0 )


    # def test_clip_pano(self, to_phi=90):
    #     panoId_2019 = "BM1Qt23drK3-yMWxYfOfVg"
    #     lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
    #     # pano1 = GSV_pano(panoId=panoId_2019, saved_path="K:\Research\street_view_depthmap")
    #     pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=os.getcwd())
    #     to_theta = 0
    #     # rimg = pano1.clip_depthmap(to_theta=to_theta, to_phi=90, zoom=3, img_type="depthmap", saved_path="K:\Research\street_view_depthmap")
    #     rimg = pano1.clip_pano(to_theta=to_theta, to_phi=-90, width=1024*2, height=768*2, zoom=4, img_type="pano", saved_path=os.getcwd())
    #     PIL.Image.fromarray(rimg).show()
    #
    #     self.assertEqual((768, 1024), rimg.shape)
        # Google link:
        # https://geo0.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&gl=us&output=thumbnail&thumb=2&w=1024&h=768&pitch=-8&ll=40.73031168738437%2C-74.18154077638651&panoid=BM1Qt23drK3-yMWxYfOfVg&yaw=194.5072326660156

     # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=os.getcwd())


   # def test_get_segmentation(self):
   #
   #      # lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
   #      # full_path = r'J:\Research\StreetView\gsv_pano\AZK1jDGIZC1zmuooSZCzEg.png'
   #      # full_path = r'D:\Code\StreetView\gsv_pano\-0D29S37SnmRq9Dju9hkqQ.png'
   #      # panoId_2019 = "-0D29S37SnmRq9Dju9hkqQ"
   #      seg_dir = r'D:\DC_segmented'
   #      pano_dir = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_panoramas'
   #      seg_files = glob.glob(os.path.join(seg_dir, "*.png"))
   #      pano_files = glob.glob(os.path.join(pano_dir, "*.jpg"))
   #      saved_path = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\test_results'
   #
   #      for idx, seg_file in enumerate(pano_files[11:12]):
   #           panoId = os.path.basename(seg_file)[:-4]
   #           pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=saved_path)
   #           # pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=r'J:\Research\StreetView\gsv_pano\test_results')
   #           pano1.set_segmentation_path(full_path=seg_file)
   #           DOM = pano1.get_DOM(width=40, height=40, resolution=0.05, zoom=4, img_type='DOM',  fill_clipped_seg=True)
   #           # palette = Image.open(seg_file).getpalette()
   #           # palette = np.array(palette,dtype=np.uint8)
   #
   #           pil_img = PIL.Image.fromarray(DOM['DOM'])
   #           # pil_img.putpalette(palette)
   #           # self.assertEqual((800, 800, 3), DOM.shape)
   #           pil_img.show()

    # def test_col_row_to_angles(self):
    #     lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
    #
    #     # pano1 = GSV_pano(panoId=panoId_2019, saved_path="D:\Code\StreetView\gsv_pano\street_view_depthmap")
    #     pano1 = GSV_pano(request_lon = lon, request_lat=lat, saved_path=r'D:\Code\StreetView\gsv_pano\test_results')
    #     zoom = 4
    #
    #
    #     image_width = pano1.jdata['Data']['level_sizes'][zoom][0][1]
    #     image_height = pano1.jdata['Data']['level_sizes'][zoom][0][0]
    #
    #     nx, ny = (image_width, image_height)
    #     x_space = np.linspace(0, nx - 1, nx)
    #     y_space = np.linspace(ny - 1, 0, ny)
    #
    #     xv, yv = np.meshgrid(x_space, y_space)
    #
    #     timer_start = time.perf_counter()
    #     thetas, phis = pano1.col_row_to_angles(xv, yv)
    #     timer_end = time.perf_counter()
    #     print("Time spent (second):", timer_end - timer_start)
    #
    #     tolerance =  math.pi / image_height * 0.5
    #     self.assertTrue(abs(thetas[0, 0] - math.pi/2) < tolerance)
    #     self.assertTrue(abs(phis[0, 0] + math.pi) < tolerance)
    #
    #     timer_start = time.perf_counter()
    #     thetas, phis = pano1.col_row_to_angles(xv.ravel(), yv.ravel())
    #     timer_end = time.perf_counter()
    #     print("Time spent (second):", timer_end - timer_start)
    #
    #     tolerance =  math.pi / image_height * 0.5
    #     self.assertTrue(abs(thetas[0] - math.pi/2) < tolerance)
    #     self.assertTrue(abs(phis[0] + math.pi) < tolerance)




    # def test_col_row_to_points(self):
    #     # lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
    #     lat, lon = 38.9484225, -77.0294996 # DC
    #     panoId_2019 = r'-0D29S37SnmRq9Dju9hkqQ'
    #     panoId_2019 = r'--rT8OYN1YM3tkQ45-dtwQ'
    #
    #     # pano1 = GSV_pano(panoId=panoId_2019, saved_path="D:\Code\StreetView\gsv_pano\street_view_depthmap")
    #     pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=r'test_results')
    #     zoom = 2
    #
    #     # pano1.get_panorama(zoom=2)
    #
    #     # pano1.get_DOM(zoom=2, img_type="DOM")
    #
    #     image_width = pano1.jdata['Data']['level_sizes'][zoom][0][1]
    #     image_height = pano1.jdata['Data']['level_sizes'][zoom][0][0]
    #
    #     nx, ny = (image_width, image_height)
    #     x_space = np.linspace(0, nx - 1, nx)
    #     y_space = np.linspace(0, ny - 1, ny)
    #
    #     xv, yv = np.meshgrid(x_space, y_space)
    #     xv = xv.astype(int)
    #     yv = yv.astype(int)
    #
    #
    #
    #     # read pixels from a image
    #     file_path = r'D:\Code\StreetView\gsv_pano\-0D29S37SnmRq9Dju9hkqQ.png'
    #     file_path = r'--rT8OYN1YM3tkQ45-dtwQ.png'
    #     # file_path = r'D:\Code\StreetView\gsv_pano\--BGis6it-dZqjEr1UR0jg.png'
    #     pil_img = Image.open(file_path)
    #     np_img = np.array(pil_img)
    #     arr_rowcol = np.argwhere(np_img > -1)
    #
    #     timer_start = time.perf_counter()
    #     #
    #     offset_row = 0
    #     offset_col =0
    #
    #
    #
    #     offset_row = 1024*2
    #     arr_col = arr_rowcol[:, 1] + offset_col
    #     arr_row = arr_rowcol[:, 0] + offset_row
    #     # P = pano1.col_row_to_points(arr_col, arr_row, zoom=zoom)
    #     P = pano1.col_row_to_points(xv.ravel(), yv.ravel(), zoom=zoom)
    #
    #     ground_mask = pano1.get_depthmap(zoom=zoom)['ground_mask']
    #     # xv = xv * ground_mask
    #     # yv = yv * ground_mask
    #     # pano1.set_segmentation_path(file_path)
    #     pixels = pano1.get_pixel_from_row_col(xv.ravel(), yv.ravel(), zoom=zoom, img_type='pano')
    #
    #     # pixels = pano1.get_depthmap(zoom=zoom)['normal_vector_map']
    #     # pixels = pixels.reshape((-1, 3))
    #
    #     # print("Len(pixels):", len(pixels))
    #     # pixels[:, 0] = np.where(pixels[:,  0] < 10, pixels[:, 0], 0)
    #     # pixels[:, 1] = np.where(pixels[:,  1] > 100, pixels[:, 1], 255)
    #     # pixels[:, 2] = np.where(pixels[:,  2] > 100, pixels[:, 1], 255)
    #     # pixels = pixels[ pixels[:,  1] > 100]
    #     # thetas, phis = pano1.col_row_to_angles(xv.ravel(), yv.ravel())
    #
    #     P = pano1.col_row_to_points(xv, yv, zoom=zoom)
    #
    #     P = np.concatenate([P, pixels], axis=1)
    #
    #
    #     distance_threshole = 30
    #     P = P[P[:, 3] < distance_threshole]
    #     # P = P[P[:, 6] > 10]
    #     # P = P[P[:, 4] < 138]
    #     # P = P[P[:, 3] > 0]
    #     timer_end = time.perf_counter()
    #     print("Time spent (second):", timer_end - timer_start)
    #
    #     v = pptk.viewer(P[:, :3])
    #     v.set(point_size=0.001, show_axis=True, show_grid=False)
    #     # set color
    #     v.attributes(P[:, 4:7] / 255.0)
    #
    #     # color_map = np.random.rand(255, 3)
    #     # v.color_map(color_map)
    #
    #     # tolerance =  math.pi / image_height * 0.5
    #     # self.assertTrue(abs(thetas[0] - math.pi/2) < tolerance)
    #     # self.assertTrue(abs(phis[0] + math.pi) < tolerance)

    # def test_get_contour(self):
    #     seg_file = r'--69cR9y-yjGxq3c-uPBRw.png'
    #     panoId = seg_file[:-4]
    #     saved_path = r'D:\Code\StreetView\gsv_pano\test_results'
    #
    #     rows_offset = 2048
    #     cols_offset = 0
    #     zoom = 4
    #
    #     pil_img = Image.open(seg_file)
    #
    #     pano1 = GSV_pano(panoId=panoId, crs_local=6487, saved_path=saved_path)
    #
    #     # DOM = pano1.get_DOM(zoom=4, resolution=0.05, fill_clipped_seg=True)
    #
    #     target_ids = [12]
    #     np_img = np.array(pil_img)
    #
    #     np_img_binary = np.zeros(np_img.shape)
    #     for i in target_ids:
    #         np_img_binary = np.logical_or(np_img_binary, np_img == i)
    #
    #     np_img_binary = np_img_binary.astype(np.uint8)
    #
    #     # cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)
    #
    #     # opened_color = cv2.merge((cv2_opened, cv2_opened, cv2_opened))
    #
    #     morph_kernel_open  = (10, 10)
    #     morph_kernel_close = (20, 20)
    #     g_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_close)
    #     g_open  = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_open)
    #
    #     cv2_img_closed = cv2.morphologyEx(np_img_binary, cv2.MORPH_CLOSE, g_close) # fill small gaps
    #     cv2_img_opened = cv2.morphologyEx(cv2_img_closed, cv2.MORPH_OPEN, g_open)
    #
    #     raw_contours, hierarchy = cv2.findContours(cv2_img_opened.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    #     cv_img_color = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    #     # cv2.drawContours(cv_img_color, raw_contours, )
    #     cv_img_con = cv2.drawContours(cv_img_color, raw_contours[18:19], -1, (0, 255, 0), 2)
    #
    #     # contours = [np.squeeze(cont) for cont in raw_contours[18:19]]
    #     contours = [np.squeeze(cont) for cont in raw_contours[:]]
    #
    #
    #     for idx, contour in enumerate(contours):
    #         cols = contour[:, 0] + cols_offset
    #         rows = contour[:, 1] + rows_offset
    #         contour_points = pano1.col_row_to_points(cols, rows, zoom=zoom)
    #         print("contour_points:", contour_points)
    #
    #         for x, y in zip(contour_points[:, 0] + 20,  20-contour_points[:, 1]):
    #             cv2.circle(cv_img_con, (int(x*30), int(y*30)), 2, (0, 255, 0), 2)
    #
    #     win_name = "opencv"
    #     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    #     cv2.moveWindow(win_name, 100, 100)
    #     cv2.imshow(win_name, cv_img_con)
    #     cv2.resizeWindow(win_name, 1600, 200)
    #     cv2.waitKey(0)



    # def test_get_road_plane(self):     # not good!
    #     panoId_2019 = r'--rT8OYN1YM3tkQ45-dtwQ'
    #
    #     timer_start = time.perf_counter()
    #
    #     pano1 = GSV_pano(panoId=panoId_2019, crs_local=6487, saved_path=r"D:\Code\StreetView\gsv_pano\test_results")
    #
    #     width=40
    #     height=40
    #     P = pano1.get_road_plane(resolution=0.05, width=width, height=height)
    #
    #     DOM = pano1.get_DOM(resolution=0.05, zoom=3, img_type="DOM")
    #     v0 = pptk.viewer(DOM['DOM_points'][:, :3])
    #     v0.set(point_size=0.001, show_axis=True, show_grid=False)
    #     v0.attributes(DOM['DOM_points'][:, 3:6]/255.0)
    #     # Image.fromarray(DOM['DOM']).show()
    #
    #     timer_end = time.perf_counter()
    #     print("Time spent (second):", timer_end - timer_start)
    #     # np_img, worldfile = pano1.points_to_DOM(P[:, 0], P[:, 1], P[:, 4:7], resolution=0.05)
    #     # new_name = os.path.join(saved_path, "points_to_image.tif")
    #     # pano1.save_image(np_img, new_name, worldfile)
    #
    #     v = pptk.viewer(P[:, :3])
    #     v.set(point_size=0.001, show_axis=True, show_grid=False)
    #     v.attributes(P[:, 3:6]/255.0)

# def test_get_ground_points(self):
    #     # lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
    #     lat, lon = 38.9484225, -77.0294996 # DC
    #     panoId_2019 = r'-0D29S37SnmRq9Dju9hkqQ'
    #     panoId_2019 = r'--rT8OYN1YM3tkQ45-dtwQ'
    #
    #     # pano1 = GSV_pano(panoId=panoId_2019, saved_path="D:\Code\StreetView\gsv_pano\street_view_depthmap")
    #     saved_path = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\test_results'
    #     pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=saved_path, crs_local=6487)
    #     zoom = 5
    #
    #     timer_start = time.perf_counter()
    #
    #     P = pano1.get_ground_points(zoom=zoom, color=True, img_type="pano")
    #
    #
    #
    #     img_w = 40
    #     img_h = 40
    #
    #     distance_threshole = img_w * 1.5
    #
    #     # P = P[P[:, 3] < distance_threshole]
    #     P = P[P[:, 0] < img_w/2]
    #     P = P[P[:, 0] > -img_w/2]
    #     P = P[P[:, 1] < img_h/2]
    #     P = P[P[:, 1] > -img_h/2]
    #
    #
    #
    #     timer_end = time.perf_counter()
    #     print("Time spent (second):", timer_end - timer_start)
    #
    #     np_img, worldfile = pano1.points_to_DOM(P[:, 0], P[:, 1], P[:, 4:7], resolution=0.05)
    #     new_name = os.path.join(saved_path, "points_to_image.tif")
    #     pano1.save_image(np_img, new_name, worldfile)
    #
    #     v = pptk.viewer(P[:, :3])
    #     v.set(point_size=0.001, show_axis=True, show_grid=False)
    #     v.attributes(P[:, 4:7]/255.0)


     # passed 2021-03-26,
     # def test_download_panorama(self):
     #      lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
     #      zoom = 1
     #      pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=r'test_results')
     #      pil_pano_img = pano1.download_panorama(zoom=zoom)
     #      pil_pano_img.show()
     #
     #      self.assertEqual(np.array(pil_pano_img).shape[:2], tuple(pano1.jdata['Data']['level_sizes'][zoom][0]))

     # passed 2021-03-26,
     # def test_get_panorama(self):
     #      lat, lon = 40.7084995,-74.2556749  # Walker Ave to Franklin elem. school, NJ
     #      zoom = 2
     #      pano1 = GSV_pano(request_lon=lon, request_lat=lat, saved_path=os.getcwd())
     #      np_pano_img = pano1.get_panorama(zoom=zoom)['image']
     #      pil_pano_img = Image.fromarray(np_pano_img)
     #      pil_pano_img.show()
     #
     #      self.assertEqual(np.array(pil_pano_img).shape[:2], tuple(pano1.jdata['Data']['level_sizes'][zoom][0]))
