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
        sidewalks = np.argwhere(predict == 11)
        sidewalks = [tuple(row) for row in sidewalks]

        sidewalks_sph = GPano.GPano.castesian_to_shperical(GPano.GPano(), sidewalks, 1024, 768, 90)
        sidewalks_sph = [tuple([row[0], row[1] + 99.08]) for row in sidewalks_sph]

        obj_json = GPano.GSV_depthmap.getJsonDepthmapfrmLonlat(GPano.GPano(), -76.741101, 38.852598)
        dm = GPano.GSV_depthmap.getDepthmapfrmJson(GPano.GSV_depthmap(), obj_json)
        pointcloud = GPano.GSV_depthmap.getPointCloud(GPano.GSV_depthmap(), sidewalks_sph, 99.08, 0, dm)
        print('pointcloud:', pointcloud)
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
