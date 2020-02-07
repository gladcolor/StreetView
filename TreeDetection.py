import cv2
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import numpy as np
import scipy.signal
from matplotlib.lines import Line2D
import math

class tree_detection():

    def __init__(self, seg_file_path, tree_label=4, clip_up=0.33, kernel_morph=15):

        try:
            self.seg_file_path = seg_file_path

            self.seg_cv2 = cv2.imread(self.seg_file_path, cv2.IMREAD_UNCHANGED)
            seg_height, seg_width = self.seg_cv2.shape
            self.seg_cv2 = self.seg_cv2[int(seg_height * clip_up):, :]  # remove the top 1/3 image.
            self.seg_cv2 = cv2.inRange(self.seg_cv2, tree_label, tree_label)  # the class lable of trees is 4
            ret, self.seg_cv2 = cv2.threshold(self.seg_cv2, 0, 1, cv2.THRESH_BINARY)

            g = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_morph, kernel_morph))

            self.closed = cv2.morphologyEx(self.seg_cv2, cv2.MORPH_CLOSE, g)
            self.opened = cv2.morphologyEx(self.seg_cv2, cv2.MORPH_OPEN, g)

            # plt.imshow(self.opened)
            # plt.show()

            self.contoursNONE, hierarchy = cv2.findContours(self.opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.contoursNONE = [np.squeeze(cont) for cont in self.contoursNONE]

            self.peaks_all = object
        except Exception as e:
            print("Error in tree_detection __init__():", e)
        # self. = object

    def getRoots(self, prominence=20, width=10, distance=20, plateau_size=(0, 100), prom_ratio=0.2):
        """

        :param prominence:
        :param width:
        :param distance:
        :param plateau_size:
        :param prom_ratio: ratio of prominence, up from root points. E.g., when prominence = 100, root point at Row # 200, \
                           prom_ratio = 0.2, the program will measure the DBH at Row # 200 -   prominence * prom_ratio  \
                           = 160
        :return: root points (cols, rows), widths of trees

        DBH: Diameter at breast height
        """

        roots_all = []
        widths = []
        for cont_num, cont in enumerate(self.contoursNONE):
            try:
                if len(cont) < 20:
                    continue

                roots = []
                print("Processing contours #:", cont_num)
                peaks_idx, dic = scipy.signal.find_peaks(cont[:, 1], prominence=prominence, width=width, distance=distance,
                                                         plateau_size=plateau_size)
                peaks = cont[peaks_idx]
                roots.append(peaks)
                roots = np.concatenate(roots)
                # roots_all.append(roots)
                print('\n', dic)

                if len(peaks_idx) == 0:
                    continue

                prominence = dic['prominences']

                DBH_row = (roots[:, 1] - prominence * prom_ratio ).astype(int)

                for idx, r in enumerate(DBH_row):

                    print('\nr:', r)

                    line_x = []
                    line_y = []

                    DBH_idx = np.argwhere(self.contoursNONE[cont_num][:, 1] == r)
                    #         print('DBH_x: ', DBH_x)
                    #         print('contoursNONE[cont_num][:, 1]:',cont_num, contoursONE[cont_num][:,:])N
                    t = [x[0] for x in DBH_idx]

                    #         print('t = [x[0] for x in DBH_idx]:', t)

                    DBH_x = self.contoursNONE[cont_num][:, 0][t]
                    #     print('DBH_x: ', DBH_x)
                    #         print('width: ', abs(DBH_x[1] - DBH_x[0]))

                    w = math.tan(math.radians(50)) * prominence[idx] / 5
                    #         print("w:", w)

                    if abs(DBH_x[0] - roots[idx, 0]) < w:

                        roots_all.append(peaks[idx])
                        widths.append(DBH_x[1] - DBH_x[0])

            except Exception as e:
                print("Error in getRoots():", e)
                continue

        return roots_all, widths

def test_getRoots():

    img_file0 = r'55103_-75.090305_40.026045_20_165'
    img_file = f'K:\\OneDrive_NJIT\\OneDrive - NJIT\\Research\\Trees\\datasets\\Philly\\Segmented_PSP\\{img_file0}.png'

    tree_detect = tree_detection(seg_file_path=img_file)

    print(tree_detect.getRoots())

if __name__ == "__main__":
    test_getRoots()