import unittest
import GPano
from PIL import Image
from skimage import io
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_castesian_to_shperical(self):
        predict = Image.open(r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\198793_17033785714_-76.741101_38.852598_0_288.png')
        predict = io.imread(
            r'O:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\streetview_images\198793_17033785714_-76.741101_38.852598_0_288.png')

        #print('predict: ', predict)
        #print('predict shape: ', predict.shape)
        predict = np.array(predict)
        sidewalks = np.argwhere(predict == 11)[:2]
        sidewalks[0] = (50, 650)   # col, row
        sidewalks[1] = (638, 290)  # col, row
        print("sidewalks[0]: ", sidewalks[0])
        sidewalks = [tuple(row) for row in sidewalks]

        print("len of sidewalks pixels: ", len(sidewalks))

        sidewalks_sph = GPano.GPano.castesian_to_shperical(GPano.GPano(), sidewalks, 1024, 768, 90)
        #sidewalks_sph = [tuple([row[0], row[1] + 99.08]) for row in sidewalks_sph]

        lon = -76.741101
        lat = 38.852598
        pano_heading = 188.92
        pano_pitch = 0.61
        pano_H = -3.285989
        thumb_heading = 288
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
        # for row in sidewalks_sph:
        #     print(row)
        dm_w = 512
        dm_h = 216
        #print(sidewalks)
        #print('predict: ', predict)
        #print('sidewalks: ', sidewalks)
        self.assertEqual(True, sidewalks_sph)


if __name__ == '__main__':
    unittest.main()
