import unittest
from GPano import *
import GPano
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import  sqlite3

class MyTestCase(unittest.TestCase):
    gpano =  object
    gsv = object
    def setUp(self) -> None:
        self.gpano = GPano.GPano()
        self.gsv = GPano.GSV_depthmap()

    def test_seg_to_landcover2(self):
        # file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\xmrpSi0qZ9UQUZKxGWMIEw_-74.2180614_40.7864947_0_53.png'
        file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\LeYAIu-xGFNEJQwOZAl3Iw_-74.209119_40.792425_0_329.16.png'
        file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\UHg76--mCi5HvNwwoPTQQw_-74.209031_40.792517_0_275.20.png'
        file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\LeYAIu-xGFNEJQwOZAl3Iw_-74.209119_40.792425_0_329.16.png'
        file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\1BV2qdirkaTkng2bqtfOMA_ - 74.205713_40.793248_0_241.75.png'
        # seglist = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024\*.png')
        seglist = glob.glob(
            r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\IEiJfDicXGTPsnQcYq0CwQ*.png')
        seglist = glob.glob(
            r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\*.png')
        # seglist = [file]
        # predicts = []
        # for seg in seglist:
        #     if not "color" in seg:
        #         predicts.append(seg)
        # #seglist = seglist[:]
        # seglist = predicts
        # seglist.append(file)

        # seglist[0] = [file]

        print("seglist:", seglist[0])
        # saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024_pc'
        saved_path = r'J:\temp\t3+pitch'
        GPano.GSV_depthmap.seg_to_landcover2(GPano.GSV_depthmap(), seglist, saved_path=saved_path, fov=math.radians(90))
        print("seg_to landcover finished.")


    # def test_colrow_to_spherial(self):
    #     theta, phi = self.gpano.colrow_to_spherial(0, 0, 1024, 768)
    #     self.assertEqual((round(0.4876162427151, 9), round(-pi/4, 9)), (round(theta, 9), round(phi, 9)))
    #
    #     theta, phi = self.gpano.colrow_to_spherial(1024, 0, 1024, 768)
    #     self.assertEqual((round(0.4876162427151, 9), round(pi/4, 9)), (round(theta, 9), round(phi, 9)))

    # def test_clip_pano(self):
    #     file = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\panos\Dg0LGX581Uu5ooKrLlvQGA.jpg'
    #     img = Image.open(file)
    #     img = np.array(img)
    #     print(self.gpano.getGSV_url_frm_panoId(os.path.basename(file)[:-4]))
    #     self.gpano.clip_pano(img, radians(0), pi/2, 1024, 768, fov=90)


    # def test_getPanoJPGfrmPanoId(self):
    #     all_files = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\json\*.json')
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\panos'
    #     for file in all_files:
    #         panoId = os.path.basename(file)[:-5]
    #         GPano.GPano.getPanoJPGfrmPanoId(GPano.GPano(), panoId, saved_path=saved_path, zoom=5)
    #         print("Finished: ", panoId)



    # def test_saveDepthMap(self):
    #     saved_path = r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\depthmap2'
    #     all_files = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\landcover\*.json')
    #     all_files = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\json\*.json')
    #     print(all_files)
    #     for file in all_files:
    #         print(file)
    #         with open(file, 'r') as f:
    #
    #             jdata = json.load(f)
    #             basename = os.path.basename(file).replace('.json', '.tif')
    #             newname = os.path.join(saved_path, basename)
    #             depthMap = GPano.GSV_depthmap.getDepthmapfrmJson(GPano.GSV_depthmap(), jdata)
    #             GPano.GSV_depthmap.saveDepthmapImage(GPano.GSV_depthmap(), depthMap, newname)
                # GPano.GPano.getPanoZoom0frmID(GPano.GPano(), basename.replace('.tif', ''), saved_path)


    # def test_get_color_pallete(self):
    #     path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\landcover\+pitch_of_thumb\*.png'
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\landcover\+pitch_of_thumb\color'
    #     all_files = glob.glob(path)
    #     for file in all_files:
    #         print(file)
    #         npimg = np.array(io.imread(file))
    #         colored = GPano.GSV_depthmap.get_color_pallete(GPano.GSV_depthmap(), npimg, 'ade20k')
    #         basename = os.path.basename(file).replace('.png', '.png')
    #         newfilename = os.path.join(saved_path, basename)
    #         colored.save(newfilename)


    # def test_seg_to_pointcloud(self):
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\xmrpSi0qZ9UQUZKxGWMIEw_-74.2180614_40.7864947_0_53.png'
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\LeYAIu-xGFNEJQwOZAl3Iw_-74.209119_40.792425_0_329.16.png'
    #
    #     seglist = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024\*.png')
    #     seglist = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\*.png')
    #     # seglist = [file]
    #     predicts = []
    #     for seg in seglist:
    #         if not "color" in seg:
    #             predicts.append(seg)
    #     #seglist = seglist[:]
    #     seglist = predicts
    #     seglist.append(file)
    #     print("seglist:", seglist[:2])
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024_pc'
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\test'
    #     GPano.GSV_depthmap.seg_to_pointcloud(GPano.GSV_depthmap(), seglist, saved_path=saved_path, fov=90)
    #     print("Finished.")

    # def test_seg_to_landcover(self):
    #     # file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\xmrpSi0qZ9UQUZKxGWMIEw_-74.2180614_40.7864947_0_53.png'
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\LeYAIu-xGFNEJQwOZAl3Iw_-74.209119_40.792425_0_329.16.png'
    #     file = r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\UHg76--mCi5HvNwwoPTQQw_-74.209031_40.792517_0_275.20.png'
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\LeYAIu-xGFNEJQwOZAl3Iw_-74.209119_40.792425_0_329.16.png'
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\1BV2qdirkaTkng2bqtfOMA_ - 74.205713_40.793248_0_241.75.png'
    #     # seglist = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024\*.png')
    #     seglist = glob.glob(
    #         r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\IEiJfDicXGTPsnQcYq0CwQ*.png')
    #     seglist = glob.glob(
    #         r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\*.png')
    #     # seglist = [file]
    #     # predicts = []
    #     # for seg in seglist:
    #     #     if not "color" in seg:
    #     #         predicts.append(seg)
    #     # #seglist = seglist[:]
    #     # seglist = predicts
    #     # seglist.append(file)
    #
    #     seglist[0] = [file]
    #
    #     print("seglist:", seglist[0])
    #     # saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024_pc'
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\test2'
    #     GPano.GSV_depthmap.seg_to_landcover(GPano.GSV_depthmap(), seglist, saved_path=saved_path, fov=90)
    #     print("seg_to landcover finished.")

    # def test_castesian_to_shperical(self):
    #     #        predict = Image.open(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\198793_17033785714_-76.741101_38.852598_0_288.png')
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\xmrpSi0qZ9UQUZKxGWMIEw_-74.2180614_40.7864947_0_53.png'
    #     #file = r'D:\\OneDrive_NJIT\\OneDrive - NJIT\\Research\\sidewalk\\Essex_test\\jpg\\segmented_1024\\-9eYImoKzQ9iOYWIPxDaXA_-74.206696_40.793733_20.2_1.1.png'
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\LeYAIu-xGFNEJQwOZAl3Iw_-74.209119_40.792425_0_329.16.png'
    #
    #     predict = io.imread(file)
    #
    #     #print('predict: ', predict)
    #     #print('predict shape: ', predict.shape)
    #     predict = np.array(predict)
    #     h, w = predict.shape
    #     sidewalks = np.argwhere((predict == 11) | (predict == 52))#[:2]
    #     #sidewalks[0] = (50, 650)   # col, row
    #     #sidewalks[1] = (638, 290)  # col, row
    #     print("sidewalks[0]: ", sidewalks[0])
    #     sidewalks = [tuple(row) for row in sidewalks]
    #
    #     # plt_x = [row[1] for row in sidewalks]
    #     # plt_y = [h - row[0] for row in sidewalks]
    #     # plt.scatter(plt_x, plt_y)
    #     # plt.show()
    #
    #     print("len of sidewalks pixels: ", len(sidewalks))
    #
    #     sidewalks_sph = GPano.GPano.castesian_to_shperical(GPano.GPano(), sidewalks, w, h, 90)
    #
    #
    #
    #     basename = os.path.basename(file)
    #     params = basename[:-4].split('_')
    #     # print("params:", params)
    #     thumb_panoId = '_'.join(params[:(len(params) - 4)])
    #     # if len(params) > 5:
    #     #     print("thumb_panoId:", thumb_panoId)
    #     # pano_lon = params[-4]
    #     # pano_lat = params[-3]
    #     # pano_heading = params[-4]
    #     # pano_pitch = params[-4]
    #     # pano_H = params[-4]
    #
    #
    #     obj_json = GPano.GPano.getJsonfrmPanoID(GPano.GPano(), thumb_panoId, dm=1)
    #     print(obj_json)
    #     #obj_json = GPano.GSV_depthmap.getJsonDepthmapfrmLonlat(GPano.GPano(), lon, lat)
    #
    #
    #     lon = -74.218032
    #     lat = 40.786579
    #     pano_heading = 9.25
    #     pano_pitch = 0.28
    #     pano_H = 15.42182
    #     thumb_heading = 53
    #     thumb_pitch = 0
    #
    #     thumb_heading = float(params[-1])
    #     thumb_pitch = float(params[-2])
    #
    #     pano_heading = obj_json["Projection"]['pano_yaw_deg']
    #     pano_heading = float(pano_heading)
    #     pano_pitch = obj_json["Projection"]['tilt_pitch_deg']
    #     pano_pitch = float(pano_pitch)
    #
    #
    #     pano_lon = float(obj_json["Location"]['original_lng'])
    #     # print('pano_lon:', pano_lon)
    #     pano_lat = float(obj_json["Location"]['original_lat'])
    #     lon = pano_lon
    #     lat = pano_lat
    #
    #
    #
    #
    #     webX, webY = GPano.GSV_depthmap.lonlat2WebMercator( GPano.GSV_depthmap(), lon, lat)
    #     print("webX, webY:", webX, webY)
    #
    #     dm = GPano.GSV_depthmap.getDepthmapfrmJson(GPano.GSV_depthmap(), obj_json)
    #     print("dm[depthmap]: ", dm['depthMap'])
    #     print("len of dm[depthmap]: ", len(dm['depthMap']))
    #     #pointcloud = GPano.GSV_depthmap.getPointCloud(GPano.GSV_depthmap(), sidewalks_sph,  thumb_heading - pano_heading, pano_pitch + thumb_pitch, dm, lon, lat, pano_H, pano_heading, pano_pitch)
    #     pointcloud = GPano.GSV_depthmap.getPointCloud(GPano.GSV_depthmap(), sidewalks_sph, thumb_heading - pano_heading,
    #                                                   thumb_pitch, dm, lon, lat, pano_H, pano_heading,
    #                                                   pano_pitch)
    #
    #     print('pointcloud:', pointcloud[0])
    #     print('pointcloud:', pointcloud[1])
    #     print(pointcloud[11])
    #     # plt_x = [row[1] for row in pointcloud]
    #     # plt_y = [row[0] for row in pointcloud]
    #     # plt.scatter(plt_x, plt_y)
    #     # plt.show()
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test'
    #     with open(os.path.join(saved_path, r'pointcloud2.csv'), 'w') as f:
    #         f.write('x,y,h,d\n')
    #         f.write('\n'.join('%s,%s,%s,%s' % x for x in pointcloud))
    #
    #
    #     with open(os.path.join(saved_path, r'sidewalk_pixels.csv'), 'w')as f:
    #         f.write('\n'.join('%s,%s,%s' % x for x in zip(sidewalks, sidewalks_sph, pointcloud)))
    #         # f.write('\n'.join('%s,%s' % x for x in sidewalks_sph))
    #         # f.write('\n'.join('%s,%s,%s,%s' % x for x in pointcloud))
    #
    #     # for row in sidewalks_sph:
    #     #     print(row)
    #     # dm_w = 512
    #     # dm_h = 216
    #     #print(sidewalks)
    #     #print('predict: ', predict)
    #     #print('sidewalks: ', sidewalks)
    #     #self.assertEqual(True, sidewalks_sph)
    #


    # def test_seg_to_pointcloud(self):
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\xmrpSi0qZ9UQUZKxGWMIEw_-74.2180614_40.7864947_0_53.png'
    #     file = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\LeYAIu-xGFNEJQwOZAl3Iw_-74.209119_40.792425_0_329.16.png'
    #     file = r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\UHg76--mCi5HvNwwoPTQQw_-74.209031_40.792517_0_275.20.png'
    #     #
    #     # seglist = glob.glob(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024\*.png')
    #     seglist = glob.glob(
    #         r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\IEiJfDicXGTPsnQcYq0CwQ*.png')
    #     seglist = glob.glob(
    #         r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0\segmented_1024\*.png')
    #     seglist = [file]
    #     # predicts = []
    #     # for seg in seglist:
    #     #     if not "color" in seg:
    #     #         predicts.append(seg)
    #     # #seglist = seglist[:]
    #     # seglist = predicts
    #     # seglist.append(file)
    #     print("seglist:", seglist)
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg\segmented_1024_pc'
    #     saved_path = r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\test\pp'
    #     GPano.GSV_depthmap.seg_to_pointcloud(GPano.GSV_depthmap(), seglist, saved_path=saved_path, fov=90)
    #     print("Finished.")


    """
    def test_getNextJson(self):
        jdata = GPano.GPano.getJsonfrmPanoID(GPano.GPano(), 'JDDtY3d4sUkNgOfTfzt1pw')
        print(jdata)
        self.assertEqual('jFXMdc6V_a1w7WrMbbmItw', GPano.GPano.getNextJson(GPano.GPano(), jdata)['Location']['panoId'])
    """

    """
    def test_getLastJson(self):
        jdata = GPano.GPano.getJsonfrmPanoID(GPano.GPano(), 'JDDtY3d4sUkNgOfTfzt1pw')
        print(jdata)
        self.assertEqual('nbz-Fu3YZlFJwZwQp-IRFA', GPano.GPano.getLastJson(GPano.GPano(), jdata)['Location']['panoId'])

    """

    """
    def test_readCoords_csv(self):
        coords = GPano.GPano.readCoords_csv(GPano.GPano(), r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\coords_of_boundary.csv')
        print(coords)
    """

    """
    def test_point_in_polygon(self):
        point = Point(-74.271202, 40.775623)
        point = Point(-64.271202, 40.775623)
        coords = GPano.GPano.readCoords_csv(GPano.GPano(),
                                            r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\coords_of_boundary.csv')
        polygon = GPano.GPano.formPolygon(GPano.GPano(), coords)
        #print(polygon)
        #print(point)
        self.assertEqual(False, GPano.GPano.point_in_polygon(GPano.GPano(), point, polygon))
    """

    """
    def test_readRoadSeedsPts_csv(self):
        pts = GPano.GPano.readRoadSeedsPts_csv(GPano.GPano(), r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\road_seeds.csv')
        print(pts)
        self.assertEqual((-74.203195, 40.794957000000004), pts[0])
    """

    """
    def test_getPanoJPGfrmArea(self):
        print('started! ')
        pts = GPano.GPano.readRoadSeedsPts_csv(GPano.GPano(),
                                               r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_county\road_seeds.csv')
        # coords = GPano.GPano.readCoords_csv(GPano.GPano(),
        #                                     r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
        coords = GPano.GPano.readCoords_csv(GPano.GPano(), r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_county\essex_vet.csv')
        polygon = GPano.GPano.formPolygon(GPano.GPano(), coords)
        saved_path = r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_county\panos'
        random.shuffle(pts)

        # self.gpano.getPanoJPGfrmArea(pts, saved_path, coords)
        self.gpano.getPanoJPGfrmArea_mp(pts, saved_path, coords, Process_cnt=10)

        # print(polygon)
        # lonlats = []
        # for pt in pts:
        #     print(pt)
        #     lonlats += (GPano.GPano.go_along_road_forward(GPano.GPano(), pt[0], pt[1],
        #                                                   saved_path=r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_county\panos',
        #                                                   yaw_list=None, pitch_list=0,
        #                                                   steps=1000000, polygon=polygon))
        #     lonlats += (GPano.GPano.go_along_road_backward(GPano.GPano(), pt[0], pt[1],
        #                                                    saved_path=r'D:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_county\panos',
        #                                                    yaw_list=None, pitch_list=0,
        #                                                    steps=1000000,
        #                                                    polygon=polygon))
        #     print("len(lonlats): ", len(lonlats))
        #
        # lons = [lonlat[0] for lonlat in lonlats]
        # lats = [lonlat[1] for lonlat in lonlats]
        #
        # plt.scatter(lons, lats)
        # plt.show()
        """

if __name__ == '__main__':
    unittest.main()
