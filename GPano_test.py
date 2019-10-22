import unittest
from GPano import *
import GPano
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

class MyTestCase(unittest.TestCase):
    """
    def test_castesian_to_shperical(self):
#        predict = Image.open(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\198793_17033785714_-76.741101_38.852598_0_288.png')
        predict = io.imread(
            r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\265746_-74.2180614_40.7864947_0_53_FR.png')

        #print('predict: ', predict)
        #print('predict shape: ', predict.shape)
        predict = np.array(predict)
        sidewalks = np.argwhere(predict == 11)#[:2]
        #sidewalks[0] = (50, 650)   # col, row
        #sidewalks[1] = (638, 290)  # col, row
        print("sidewalks[0]: ", sidewalks[0])
        sidewalks = [tuple(row) for row in sidewalks]

        print("len of sidewalks pixels: ", len(sidewalks))

        sidewalks_sph = GPano.GPano.castesian_to_shperical(GPano.GPano(), sidewalks, 512, 384, 90)
        #sidewalks_sph = [tuple([row[0], row[1] + 99.08]) for row in sidewalks_sph]

        lon = -74.218032
        lat = 40.786579
        pano_heading = 9.25
        pano_pitch = 0.28
        pano_H = 15.42182
        thumb_heading = 53
        thumb_pitch = 0

        obj_json = GPano.GSV_depthmap.getJsonDepthmapfrmLonlat(GPano.GPano(), lon, lat)

        webX, webY = GPano.GSV_depthmap.lonlat2WebMercator( GPano.GSV_depthmap(), lon, lat)
        print("webX, webY:", webX, webY)

        dm = GPano.GSV_depthmap.getDepthmapfrmJson(GPano.GSV_depthmap(), obj_json)
        print("dm[depthmap]: ", dm['depthMap'])
        print("len of dm[depthmap]: ", len(dm['depthMap']))
        pointcloud = GPano.GSV_depthmap.getPointCloud(GPano.GSV_depthmap(), sidewalks_sph,  thumb_heading - pano_heading, pano_pitch + thumb_pitch, dm, lon, lat, pano_H, pano_heading, pano_pitch)
        print('pointcloud:', pointcloud[0])
        print('pointcloud:', pointcloud[1])
        print(pointcloud[11])
        plt_x = [row[0] for row in pointcloud]
        plt_y = [row[1] for row in pointcloud]
        plt.scatter(plt_x, plt_y)
        plt.show()
        with open(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\pointcloud.csv', 'w') as f:
            f.write('\n'.join('%s,%s,%s,%s' % x for x in pointcloud))

        with open(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\sidewalk_pixels.csv', 'w') as ff:
            ff.write('\n'.join('%s,%s' % x for x in sidewalks))

        # for row in sidewalks_sph:
        #     print(row)
        dm_w = 512
        dm_h = 216
        #print(sidewalks)
        #print('predict: ', predict)
        #print('sidewalks: ', sidewalks)
        self.assertEqual(True, sidewalks_sph)

"""

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

    def test_go_along_road_forward(self):
        pts = GPano.GPano.readRoadSeedsPts_csv(GPano.GPano(),
                                               r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\road_seeds.csv')
        coords = GPano.GPano.readCoords_csv(GPano.GPano(),
                                            r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\polygon_coords.csv')
        polygon = GPano.GPano.formPolygon(GPano.GPano(), coords)

        print(polygon)
        lonlats = []
        for pt in pts:
            print(pt)
            lonlats += (GPano.GPano.go_along_road_forward(GPano.GPano(), pt[0], pt[1], saved_path='', steps=1000000, polygon=polygon))
            lonlats += (GPano.GPano.go_along_road_backward(GPano.GPano(), pt[0], pt[1], saved_path='', steps=1000000,
                                                          polygon=polygon))
            print(len(lonlats))

        lons = [lonlat[0] for lonlat in lonlats]
        lats = [lonlat[1] for lonlat in lonlats]

        plt.scatter(lons, lats)
        plt.show()
if __name__ == '__main__':
   unittest.main()
