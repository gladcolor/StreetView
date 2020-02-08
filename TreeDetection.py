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

    def __init__(self, seg_file_path, tree_label=4, clip_up=0.5, kernel_morph=8, kernel_list=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120]):

        try:
            self.seg_file_path = seg_file_path

            self.seg_cv2 = cv2.imread(self.seg_file_path, cv2.IMREAD_UNCHANGED)

            self.seg_height, self.seg_width = self.seg_cv2.shape

            self.seg_cv2 = self.seg_cv2[int(self.seg_height * clip_up):, :]  # remove the top 1/3 image.

            self.seg_height, self.seg_width = self.seg_cv2.shape

            self.seg_cv2 = cv2.inRange(self.seg_cv2, tree_label, tree_label)  # the class lable of trees is 4
            ret, self.seg_cv2 = cv2.threshold(self.seg_cv2, 0, 1, cv2.THRESH_BINARY)

            g = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_morph, kernel_morph))

            self.closed = cv2.morphologyEx(self.seg_cv2, cv2.MORPH_CLOSE, g)
            self.opened = cv2.morphologyEx(self.seg_cv2, cv2.MORPH_OPEN, g)

            self.sobel_v = cv2.Sobel(self.opened, cv2.CV_64F, 1, 0, ksize=3)
            self.sobel_v_abs = np.abs(self.sobel_v)
            self.sobel_v_abs = np.where(self.sobel_v_abs > 2.0, self.sobel_v_abs, 0)
            self.sobel_v_abs[:, :-2] = 1 # set the edge to sobel_v
            self.sobel_v_abs[:, 0:3] = 1  # set the edge to sobel_v

            self.sobel_h = cv2.Sobel(self.opened, cv2.CV_64F, 0, 1, ksize=3)
            self.sobel_h_abs = np.abs(self.sobel_h)
            self.sobel_h_abs = np.where(self.sobel_h_abs > 2.0, self.sobel_h_abs, 0)

            # print((sobel_v))
            # fig = plt.figure(figsize=(10, 8))
            # plt.imshow(sobel_v_abs)
            # plt.title("sobel_v_abs")
            # plt.colorbar()
            # plt.show()

            # fig = plt.figure(figsize=(10, 8))
            # plt.imshow(sobel_h)
            # plt.title("sobel_h")
            # plt.colorbar()
            # plt.show()

            self.contoursNONE, hierarchy = cv2.findContours(self.opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # con = cv2.drawContours(self.opened, self.contoursNONE, -1, (255, 255, 100), 3)
            #
            # plt.imshow(con)
            # plt.show()

            self.contoursNONE = [np.squeeze(cont) for cont in self.contoursNONE]

            self.kernel_list = kernel_list

        except Exception as e:
            print("Error in tree_detection __init__():", e)
        # self. = object

    def conv_verify(self, row, col):
        sum_conv = 0

        for kernel in self.kernel_list:
            kernel_w = kernel
            kernel_h = kernel_w * 1.5
            threshold = kernel_h * 1.5
            #         print(threshold)
            if (row > kernel_h) and (col > kernel_w / 2):
                if (row < self.seg_height) and (col < (self.seg_width - kernel_w / 2)):
                    conved = self.sobel_v_abs[int(row - kernel_h):int(row), int(col - kernel_w / 2):int(col + kernel_w / 2)]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2

                    if sum_conv > threshold:
                        if self.sobel_h[row, col] < -1:
                            return True
                    #

                    # return sum_conv, kernel_w

        return False

    def getRoots(self, prominence=30, width=10, distance=20, plateau_size=(0, 150), prom_ratio=0.2):
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
        isDraw = 1

        if isDraw:
            fig, ax = plt.subplots()
            plt.imshow(self.opened)

        roots_all = []
        widths = []
        # self.contoursNONE = self.contoursNONE[2:]

        verifieds_x = []
        verifieds_y = []

        for cont_num, cont in enumerate(self.contoursNONE):
            try:
                if len(cont) < 40:
                    continue

                roots = []
                print("Processing contours #:", cont_num)

                peaks_idx, dic = scipy.signal.find_peaks(cont[:, 1], prominence=prominence, width=width, distance=distance,
                                                         plateau_size=plateau_size)
                # print("cont:", cont)
                # peaks_idx, dic = scipy.signal.find_peaks(cont[:, 1], prominence=10, width=10, distance=20,
                #                                          plateau_size=10)

                peaks = cont[peaks_idx]
                roots.append(peaks)
                roots = np.concatenate(roots)
                # roots_all.append(roots)
                # print('\n', dic)

                if len(peaks_idx) == 0:
                    continue

                prominences = dic['prominences']

                DBH_row = (roots[:, 1] - prominences * prom_ratio).astype(int)
                DBH_row = (roots[:, 1] - prominence).astype(int)

                ax.scatter(peaks[:, 0], peaks[:, 1], color='red', s=50)
                # plt.show()

                for peak in peaks:
                    col = peak[0]
                    row = peak[1]

                    if self.conv_verify(row, col):
                        verifieds_x.append(col)
                        verifieds_y.append(row)
                        print("Verified!", col, row)



                for idx, r in enumerate(DBH_row):

                    # print('\nr:', r)

                    line_x = []
                    line_y = []

                    DBH_idx = np.argwhere(self.contoursNONE[cont_num][:, 1] == r)

                    t = [x[0] for x in DBH_idx]


                    DBH_x = self.contoursNONE[cont_num][:, 0][t]

                    # remove the adjacent pixels
                    temp = [DBH_x[0]]
                    for x in range(1, len(DBH_x)):
                        if (DBH_x[x] - DBH_x[x - 1]) > 1:
                            temp.append(DBH_x[x])
                    DBH_x = temp


                    print('DBH_x: ', DBH_x)
                    #         print('width: ', abs(DBH_x[1] - DBH_x[0]))

                    w = math.tan(math.radians(40)) * prominences[idx] * prom_ratio
                    w = math.tan(math.radians(40)) * prominences[idx] * prom_ratio
                    #         print("w:", w)



                    if abs(DBH_x[0] - roots[idx, 0]) < 80:
                        roots_all.append(peaks[idx])
                        widths.append(DBH_x[1] - DBH_x[0])
                        line_x.append(DBH_x[1])
                        line_x.append(DBH_x[0])
                        line_y.append(r)
                        line_y.append(r)

                        if isDraw:
                            ax.add_line(Line2D(line_x, line_y, color='r'))
            except Exception as e:
                print("Error in getRoots():", e)
                continue

        ax.scatter(verifieds_x, verifieds_y, color='green', s=20)
        plt.show()
        # plt.show()

        if len(roots_all) > 0:
            roots_all = np.concatenate(roots_all).reshape((len(widths), -1))

            if isDraw:   # draw results
                ax.scatter(roots_all[:, 0], roots_all[:, 1], color='red', s=12)
                plt.show()

        return roots_all, widths

def test_getRoots():

    img_file0 = r'55019_-75.074928_40.028516_20_27' # failed
    img_file0 = r'56781_-75.116041_40.027007_20_353' # failed
    img_file0 = r'56666_-75.135652_39.978263_20_102'
    img_file0 = r'56615_-75.135871_39.977264_20_148'
    img_file0 = r'56507_-75.142394_40.007453_20_134'
    img_file0 = r'56436_-75.139345_40.019191_20_223'
    img_file0 = r'56431_-75.139869_40.016782_20_305'
    img_file0 = r'56389_-75.134487_39.993348_20_227'
    img_file0 = r'56323_-75.138984_40.010789_20_237'
    img_file0 = r'56279_-75.137075_39.981078_20_79'


    img_file = f'K:\\OneDrive_NJIT\\OneDrive - NJIT\\Research\\Trees\\datasets\\Philly\\Segmented_PSP\\{img_file0}.png'

    tree_detect = tree_detection(seg_file_path=img_file)
    roots_all, widths = tree_detect.getRoots()
    print(roots_all, widths)


    # plt.imshow(tree_detect.opened)
    # plt.scatter(roots_all[:, 0], roots_all[:, 1])
    # plt.show()


if __name__ == "__main__":
    test_getRoots()