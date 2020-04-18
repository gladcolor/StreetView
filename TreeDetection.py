import cv2
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import numpy as np
import scipy.signal
from matplotlib.lines import Line2D
import math
import os
import glob
from GPano import GPano
from GPano import GSV_depthmap
from math import *
import multiprocessing as mp

GSV_FOLDER = r''
SEG_FOLDER = r''
COLOR_FOLDER = r''

SEG_FOLDER = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\datasets\Ottawa\tree_seg\*.png'
GSV_FOLDER = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\datasets\Ottawa\tree_jpg2'
COLOR_FOLDER = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\datasets\Ottawa\tree_color'
SAVED_FOLDER = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\datasets\Ottawa\tree_detected2'
SAVED_FILE = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\datasets\Ottawa\Trees_img_only.csv'

class tree_detection():

    def __init__(self, seg_file_path, tree_label=4, clip_up=0.4, kernel_morph=3, kernel_list=[5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150, 180, 200]):

        try:
            self.seg_file_path = seg_file_path
            self.clip_up = clip_up

            self.seg_cv2 = cv2.imread(self.seg_file_path, cv2.IMREAD_UNCHANGED)

            self.seg_height, self.seg_width = self.seg_cv2.shape

            self.seg_cv2 = self.seg_cv2[int(self.seg_height * clip_up):, :]  # remove the top 1/3 image.

            # self.gsv_folder = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\sidewalk\Essex_test\jpg_pitch0'
            # self.img_gsv_filename = os.path.join(self.gsv_folder, os.path.basename(seg_file_path.replace('.png', '.jpg')))

            # global SEG_FOLDER  # = r'J:\Research\Trees\west_trees_seg\*.png'
            # global GSV_FOLDER  # = r'J:\Research\Trees\west_trees'
            # global COLOR_FOLDER  # = r'J:\Research\Trees\west_trees_color'
            # global SAVED_FOLDER  # = r'J:\Research\Trees\west_trees_detected'


            self.gsv_folder = GSV_FOLDER
            self.seg_folder = SEG_FOLDER
            self.color_folder = COLOR_FOLDER

            # read jpg files.
            # self.img_gsv_filename = os.path.join(self.gsv_folder,
            #                                      os.path.basename(seg_file_path.replace('.png', '.jpg')))
            # self.img_gsv_cv2 = cv2.imread(self.img_gsv_filename, cv2.IMREAD_UNCHANGED)

            # read png files.

            if self.color_folder != "":
                self.img_color_filename = os.path.join(self.color_folder,
                                                 os.path.basename(seg_file_path.replace('.png', '_color.png')))
                self.img_color_cv2 = cv2.imread(self.img_color_filename)

            if self.gsv_folder != "":
                self.img_gsv_filename = os.path.join(self.gsv_folder,
                                                     os.path.basename(seg_file_path.replace('.png', '.jpg')))
                self.img_gsv_cv2 = cv2.imread(self.img_gsv_filename)


            self.gsv_height, self.gsv_width, self.gsv_channels = self.img_gsv_cv2.shape
            # self.img_gsv_cv2 = self.img_gsv_cv2[int(self.seg_height * clip_up):, :]
            self.img_gsv_cv2 = cv2.cvtColor(self.img_gsv_cv2, cv2.COLOR_BGR2RGB)

            # self.seg_height, self.seg_width = self.seg_cv2.shape

            self.seg_cv2 = cv2.inRange(self.seg_cv2, tree_label, tree_label)  # the class lable of trees is 4
            ret, self.seg_cv2 = cv2.threshold(self.seg_cv2, 0, 1, cv2.THRESH_BINARY)

            g = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_morph, kernel_morph))

            self.closed = cv2.morphologyEx(self.seg_cv2, cv2.MORPH_CLOSE, g)
            self.opened = cv2.morphologyEx(self.seg_cv2, cv2.MORPH_OPEN, g)

            self.sobel_v = cv2.Sobel(self.opened, cv2.CV_64F, 1, 0, ksize=3)
            # plt.imshow(self.sobel_v)
            # plt.title("sobel_v")
            # plt.colorbar()
            # plt.show()

            self.sobel_v_abs = np.abs(self.sobel_v)
            # plt.imshow(self.sobel_v_abs)
            # plt.title("sobel_v_abs")
            # plt.colorbar()
            # plt.show()

            self.sobel_v_abs = np.where(self.sobel_v_abs > 2.0, self.sobel_v_abs, 0)
            self.sobel_v_abs[:, 0:1] = 1 # set the left and right columns as vertical edge
            self.sobel_v_abs[:, -1:] = 1  # set the edge to sobel_v
            # self.sobel_v_abs[0, :] = 10  # set the edge to sobel_v
            # self.sobel_v_abs[-1, :] = 10  # set the edge to sobel_v
            self.sobel_v_bin = np.where(self.sobel_v_abs > 0, 1, 0)

            # plt.imshow(self.sobel_v_bin)
            # plt.title("sobel_v_bin")
            # plt.colorbar()
            # plt.show()

            self.sobel_h = cv2.Sobel(self.opened, cv2.CV_64F, 0, 1, ksize=3)
            self.sobel_h_abs = np.abs(self.sobel_h)
            self.sobel_h_abs = np.where(self.sobel_h_abs > 2.0, self.sobel_h_abs, 0)

            # print((sobel_v))
            # fig = plt.figure(figsize=(20, 13))
            # plt.imshow(self.sobel_v_abs)
            # plt.title("sobel_v_abs final ")
            # plt.colorbar()
            # plt.show()

            # fig = plt.figure(figsize=(10, 8))
            # plt.imshow(sobel_h)
            # plt.title("sobel_h")
            # plt.colorbar()
            # plt.show()

            self.contoursNONE, hierarchy = cv2.findContours(self.opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            con = cv2.drawContours(self.opened, self.contoursNONE, -1, (255, 255, 100), 3)


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
            ratio = 1.5
            if kernel < 20:
                ratio = 1.5
            kernel_h = kernel_w * ratio
            threshold = kernel_h * ratio
            #         print(threshold)


            # print(f"Conv_verify(), left: {left}, right: {right}, up: {up}, bottom: {bottom}")

            if (row > kernel_h) and (col > kernel_w / 2):
                if col < (self.seg_width - kernel_w / 2):
                    # conved = self.sobel_v_abs[up: bottom, left: right]

                    # the root is in the middle
                    left = max(0, int(col - kernel_w / 2))
                    right = min(self.seg_width, int(col + kernel_w / 2))
                    up = max(0, int(row - kernel_h))
                    bottom = row

                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2 # the edge is 2 pixel width

                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")
                    if sum_conv > threshold:
                        if sobel_h < -1:    # if in the maximum point
                            return True, kernel_w

                    # the root is in the left
                    left = col
                    right =col + kernel_w
                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2 # the edge is 2 pixel width
                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")

                    if sum_conv > threshold:
                        if sobel_h < -1:
                            return True, kernel_w

                    # the root is in the right
                    left = col - kernel_w
                    right =col
                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2  # the edge is 2 pixel width

                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")

                    if sum_conv > threshold:
                        if sobel_h < -1:
                            return True, kernel_w



        return False, 0

    def conv_verify(self, row, col):
        sum_conv = 0

        for kernel in self.kernel_list:
            kernel_w = kernel
            ratio = 1.5
            if kernel < 20:
                ratio = 1.5
            kernel_h = kernel_w * ratio
            threshold = kernel_h * ratio
            #         print(threshold)


            # print(f"Conv_verify(), left: {left}, right: {right}, up: {up}, bottom: {bottom}")

            if (row > kernel_h) and (col > kernel_w / 2):
                if col < (self.seg_width - kernel_w / 2):
                    # conved = self.sobel_v_abs[up: bottom, left: right]

                    # the root is in the middle
                    left = max(0, int(col - kernel_w / 2))
                    right = min(self.seg_width, int(col + kernel_w / 2))
                    up = max(0, int(row - kernel_h))
                    bottom = row

                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2 # the edge is 2 pixel width

                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")
                    if sum_conv > threshold:
                        if sobel_h < -1:    # if in the maximum point
                            return True, kernel_w

                    # the root is in the left
                    left = col
                    right =col + kernel_w
                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2 # the edge is 2 pixel width
                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")

                    if sum_conv > threshold:
                        if sobel_h < -1:
                            return True, kernel_w

                    # the root is in the right
                    left = col - kernel_w
                    right =col
                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2  # the edge is 2 pixel width

                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")

                    if sum_conv > threshold:
                        if sobel_h < -1:
                            return True, kernel_w



        return False, 0

    def conv_verify(self, row, col):
        sum_conv = 0

        for kernel in self.kernel_list:
            kernel_w = kernel
            ratio = 1.5
            if kernel < 20:
                ratio = 1.5
            kernel_h = kernel_w * ratio
            threshold = kernel_h * ratio
            #         print(threshold)


            # print(f"Conv_verify(), left: {left}, right: {right}, up: {up}, bottom: {bottom}")

            if (row > kernel_h) and (col > kernel_w / 2):
                if col < (self.seg_width - kernel_w / 2):
                    # conved = self.sobel_v_abs[up: bottom, left: right]

                    # the root is in the middle
                    left = max(0, int(col - kernel_w / 2))
                    right = min(self.seg_width, int(col + kernel_w / 2))
                    up = max(0, int(row - kernel_h))
                    bottom = row

                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2 # the edge is 2 pixel width

                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")
                    if sum_conv > threshold:
                        if sobel_h < -1:    # if in the maximum point
                            return True, kernel_w

                    # the root is in the left
                    left = col
                    right =col + kernel_w
                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2 # the edge is 2 pixel width
                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")

                    if sum_conv > threshold:
                        if sobel_h < -1:
                            return True, kernel_w

                    # the root is in the right
                    left = col - kernel_w
                    right =col
                    conved = self.sobel_v_bin[up: bottom, left: right]
                    conved = np.where(conved > 0, 1, 0)
                    sum_conv = np.sum(conved) / 2  # the edge is 2 pixel width

                    sobel_h = self.sobel_h[row, col]
                    print(f"Conv_verify(), col: {col}, row: {row}, sobel_h: {sobel_h}")
                    print(f"Conv_verify(), kernel: {kernel}, threshold: {threshold}, sum_conv: {sum_conv}")

                    if sum_conv > threshold:
                        if sobel_h < -1:
                            return True, kernel_w



        return False, 0


    def conv_verify_simplify(self, row, col, simplified_contour):
        try:
            for kernel in self.kernel_list:  # build a set of boxes, if the peak-point match one box, it is a root.
                kernel_w = kernel
                ratio = 1.5
                if kernel > 50:
                    ratio = 1
                if kernel > 50:
                    ratio = 0.7

                kernel_h = kernel_w * ratio
                # threshold = kernel_h  #* ratio
                #         print(threshold)

                # if the root is in the middle
                left = max(0, int(col - kernel_w / 1))
                right = min(self.seg_width-1, int(col + kernel_w / 1))
                up = max(0, int(row - kernel_h))
                bottom = row

                # check the left side
                lowest_row = simplified_contour[left, 1]
                lowest_row = max(lowest_row, up)
                left_height = bottom - lowest_row + 1

                # check the right side
                lowest_row = simplified_contour[right, 1]
                lowest_row = max(lowest_row, up)
                right_height = bottom - lowest_row + 1

                if (left_height > kernel_h) and (right_height > kernel_h):
                    # find the width
                    r_lip = col
                    l_lip = col

                    while simplified_contour[r_lip, 1] > up:
                        r_lip += 1
                    while simplified_contour[l_lip, 1] > up:
                        l_lip -= 1
                    width = r_lip - l_lip
                    return True, width

            return False, 0
        except Exception as e:
            print("Error in conv_verify_simplify(): ", e)

    def conv_verify0(self, row, col):  # original, works.
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
                        # if self.sobel_h[row, col] < -1:
                        return True
        return False

    def simplify_contour(self, contour, width = 1024):
        # width: the width of the image

        left = contour[:, 0].min()
        right = contour[:, 0].max()
        x_range = right - left
        simplified_con = np.zeros((width, 2))

        min_y = contour[:, 1].min()

        for x in range(left, right):
            simplified_con[x, 1] = contour[contour[:, 0] == x][:, 1].max()
            simplified_con[x, 0] = x

        for x in range(0, left):
            simplified_con[x, 1] = min_y
            simplified_con[x, 0] = x

        for x in range(right, width):
            simplified_con[x, 1] = min_y
            simplified_con[x, 0] = x

        return simplified_con.astype(int)


    # not simplify the contour
    def getRoots0(self, prominence=20, width=10, distance=20, plateau_size=(0, 150), prom_ratio=0.25):
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
            fig = plt.figure(figsize=(24, 14))
            plt.title(os.path.basename(self.seg_file_path))
            ax_gsv = fig.add_subplot(121)
            ax_color = fig.add_subplot(122)
            ax_gsv.imshow(self.img_gsv_cv2)
            # ax_color.imshow(self.img_color_cv2)
            ax_color.imshow(self.sobel_v_bin)
            plt.tight_layout()
            # plt.show()

        roots_all = []
        widths = []
        # self.contoursNONE = self.contoursNONE[2:]

        verifieds_x = []
        verifieds_y = []

        try:

            if len(self.contoursNONE) < 1:   # no tree is found
                return roots_all, widths
        except:
            return  roots_all, widths

        for cont_num, cont in enumerate(self.contoursNONE):  # for each contour from openCV2
            try:
                if len(cont) < 40:    # ignore small contour (i.g. polygon)
                    continue

                roots = []
                print("Processing contours #:", cont_num)

                # find peaks (i.g. roots).
                peaks_idx, dic = scipy.signal.find_peaks(cont[:, 1], prominence=prominence, width=width, distance=distance,
                                                         plateau_size=plateau_size)


                # print("cont:", cont)
                # peaks_idx, dic = scipy.signal.find_peaks(cont[:, 1], prominence=10, width=10, distance=20,
                #                                          plateau_size=10)

                # simplified_con = self.simplify_contour(cont, )


                peaks = cont[peaks_idx]   # get the col / row of peak

                if isDraw:  # draw peaks
                    # ax.scatter(roots_all[:, 0], roots_all[:, 1], color='red', s=12)
                    ax_color.scatter(peaks[:,0], peaks[:,1]+ self.gsv_height * self.clip_up, color='cyan', s=40)
                    ax_gsv.scatter(peaks[:,0], peaks[:,1]+ self.gsv_height * self.clip_up, color='cyan', s=40)

                verifieds_idx = []
                verified_kernels = []
                for i, peak in enumerate(peaks):
                    col = peak[0]
                    row = peak[1]
                    verified, kernel = self.conv_verify(row, col)
                    if verified:    # check whether the peak fits to requirements
                        verifieds_x.append(col)
                        verifieds_y.append(row + self.gsv_height * self.clip_up)
                        verifieds_idx.append(i)
                        verified_kernels.append(kernel)
                        # print("Verified!", col, row)

                peaks = peaks[verifieds_idx]  # store the roots only
                roots.append(peaks)
                roots = np.concatenate(roots)
                # roots_all.append(roots)
                # print('\n', dic)

                if len(peaks_idx) == 0:   # if no roots
                    continue

                prominences = dic['prominences']   # height of peak (i.g. root)

                # DBH_row = (roots[:, 1] - prominences[verifieds_idx] * prom_ratio).astype(int)  # measure dimater in these rows
                DBH_row = (roots[:, 1] - verified_kernels).astype(int)

                #peaks_widths = dic['peaks_widths']
                #peaks_prominences = dic['prominences']

                # if isDraw:
                #     # ax.hlines(y=dic["width_heights"] + self.gsv_height * self.clip_up, xmin=dic["left_ips"], xmax=dic["right_ips"], color="C1")
                #
                #     ax.scatter(peaks[:,0], peaks[:,1], color='red', s=50)
                #plt.show()


                for idx, r in enumerate(DBH_row):  # measure each trunk

                    # print('\nr:', r)
                    root_x = verifieds_x[idx]
                    kernel = verified_kernels[idx]
                    # store the left & right ends for drawing widths.
                    line_x = []
                    line_y = []

                    # get the columns of possible left & right ends
                    DBH_idx = np.argwhere(self.contoursNONE[cont_num][:, 1] == r)    # get the nodes in this r (row).
                    t = [x[0] for x in DBH_idx]  # convert it to 1-D list.
                    DBH_x = self.contoursNONE[cont_num][:, 0][t]  # get cols of these nodes.

                    # remove the columns beyond the kernel
                    temp = []  # t
                    for x in DBH_x:
                        if (x >= (root_x - kernel)) and (x <=(root_x + kernel)):
                            temp.append(x)
                    DBH_x = temp

                    # remove the adjacent pixels
                    temp = [DBH_x[0]]  # the most left col
                    for x in range(1, len(DBH_x)):   # find the right col
                        if (DBH_x[x] - DBH_x[x - 1]) > 1:
                            temp.append(DBH_x[x])
                    DBH_x = temp


                    # print('DBH_x: ', DBH_x)
                    #         print('width: ', abs(DBH_x[1] - DBH_x[0]))
                    # w = math.tan(math.radians(40)) * prominences[idx] * prom_ratio
                    #         print("w:", w)

                    MAX_DBH = 180 # maximum DBH, pixels
                    # if abs(roots[idx, 0] - DBH_x[0]) < (MAX_DBH / 2):   # remove the left edge of trunk is more than 80 pixels, ignore it.
                    if (DBH_x[-1] - DBH_x[0]) < MAX_DBH:  # remove the left edge of trunk is more than 80 pixels, ignore it.

                        roots_all.append(peaks[idx])
                        widths.append(DBH_x[1] - DBH_x[0])
                        line_x.append(DBH_x[1])
                        line_x.append(DBH_x[0])
                        line_y.append(r + self.gsv_height * self.clip_up)
                        line_y.append(r + self.gsv_height * self.clip_up)

                        if isDraw:
                            ax_color.add_line(Line2D(line_x, line_y, color='r'))
                            ax_gsv.add_line(Line2D(line_x, line_y, color='r'))
                            
            except Exception as e:
                print("Error in getRoots0():", e)
                continue

        if isDraw:  # draw results
            # ax.scatter(roots_all[:, 0], roots_all[:, 1], color='red', s=12)
            ax_color.scatter(verifieds_x, verifieds_y, color='red', s=40, marker='+')
            ax_gsv.scatter(verifieds_x, verifieds_y, color='red', s=40)

            for idx, x in enumerate(verifieds_x):
                kernel = verified_kernels[idx]
                line_x = [x - kernel / 2, x + kernel / 2]
                line_y = [verifieds_y[idx], verifieds_y[idx]]
                ax_color.add_line(Line2D(line_x, line_y, color='b'))


            plt.savefig(os.path.join(SAVED_FOLDER, \
                                     os.path.basename(self.seg_file_path).replace(".png", ".png")))
            # plt.show()
            # fig.clf()
            # plt.clf(fig)
            plt.close('all')
            #

        if len(roots_all) > 0:
            roots_all = np.concatenate(roots_all).reshape((len(widths), -1))
            roots_all[:, 1] = roots_all[:, 1] + self.gsv_height * self.clip_up
            # plt.show()


        return roots_all, widths

    #  simplify the contour
    def getRoots_simplified(self, prominence=20, width=10, distance=20, plateau_size=(0, 150), prom_ratio=0.25):
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
            fig = plt.figure(figsize=(24, 14))
            plt.title(os.path.basename(self.seg_file_path))
            ax_gsv = fig.add_subplot(121)
            ax_color = fig.add_subplot(122)
            ax_gsv.imshow(self.img_gsv_cv2)
            ax_color.imshow(self.img_color_cv2)
            # ax_color.imshow(self.sobel_v_bin)
            # plt.tight_layout()
            # plt.show()

        # return these values
        roots_all = []
        widths = []
        DBH_points = []


        try:
            if len(self.contoursNONE) < 1:  # no tree is found
                return roots_all, widths
        except:
            print("Not countour. \n")
            return roots_all, widths, DBH_points

        for cont_num, cont in enumerate(self.contoursNONE):  # for each contour from openCV2
            try:
                verifieds_x = []
                verifieds_y = []
                if len(cont) < 200:  # ignore small contour (i.g. polygon)
                    continue

                roots = []
                print("Processing contours #:", cont_num)

                simplified_con = self.simplify_contour(cont)

                peaks_idx, dic = scipy.signal.find_peaks(simplified_con[:, 1], prominence=prominence, width=width,
                                                         distance=distance,
                                                         plateau_size=plateau_size)
                print(f"Found {len(peaks_idx)} peaks at: {peaks_idx} . \n")

                peaks_simplified = simplified_con[peaks_idx]  # get the col / row of peak

                if isDraw:  # draw peaks
                    # ax.scatter(roots_all[:, 0], roots_all[:, 1], color='red', s=12)

                    ax_gsv.scatter(  peaks_simplified[:, 0], peaks_simplified[:, 1] + self.gsv_height * self.clip_up, color='cyan', s=40)
                    ax_color.scatter(peaks_simplified[:, 0], peaks_simplified[:, 1] + self.gsv_height * self.clip_up, color='cyan', s=40)
                    # ax_color.plot(simplified_con)

                    properties = dic
                    x = simplified_con[:, 1] +  self.gsv_height * self.clip_up
                    # ax_color.scatter(simplified_con)
                    xx = x
                    min_in_xx = xx.min()
                    # np.minimum(xx, min_in_xx)
                    xx[xx == min_in_xx] = 0
                    # ax_color.plot(xx)
                    ax_color.plot(peaks_idx, x[peaks_idx], "x")
                    # ax_color.vlines(x=peaks_idx, ymin=x[peaks_idx] - properties["prominences"], ymax = x[peaks_idx], color = "C1")
                    # ax_color.hlines(y=properties["width_heights"] + self.gsv_height * self.clip_up, xmin=properties["left_ips"], xmax = properties["right_ips"], color = "C1")

                verifieds_idx = []
                verified_kernels = []
                for i, peak in enumerate(peaks_simplified):
                    col = peak[0]
                    row = peak[1]
                    verified, kernel = self.conv_verify_simplify(row, col, simplified_con)
                    if verified:  # check whether the peak fits to requirements
                        verifieds_x.append(col)
                        verifieds_y.append(row + self.gsv_height * self.clip_up)
                        verifieds_idx.append(i)
                        verified_kernels.append(kernel)
                        roots_all.append(peak)

                peaks = peaks_simplified[verifieds_idx]  # store the roots only
                roots.append(peaks)
                roots = np.concatenate(roots)

                print('roots: ', roots)

                if len(peaks_idx) == 0:  # if no roots
                    continue

                if isDraw:  # draw results
                    for idx, x in enumerate(verifieds_x):
                        kernel = verified_kernels[idx]
                        # pass
                        line_x = [x - kernel / 2, x + kernel / 2]
                        line_y = [verifieds_y[idx], verifieds_y[idx]]
                        ax_color.add_line(Line2D(line_x, line_y, color='red'))
                        ax_color.scatter(verifieds_x, verifieds_y, color='y', s=40, marker='+')
                        ax_gsv.scatter(verifieds_x, verifieds_y, color='red', s=40)

            except Exception as e:
                print("Error in getRoots0():", e)
                continue



        if isDraw:  # draw results
            try:
                # ax.scatter(roots_all[:, 0], roots_all[:, 1], color='red', s=12)

                plt.savefig(os.path.join(SAVED_FOLDER, \
                                         os.path.basename(self.seg_file_path).replace(".png", ".png")))
                # plt.show()
                # fig.clf()
                # plt.clf(fig)
                plt.close('all')
            except Exception as e:
                print("Error in Drawing:", e)
            #

        if len(roots_all) > 0:
            roots_all = np.concatenate(roots_all).reshape((len(roots_all), -1))
            roots_all[:, 1] = roots_all[:, 1] + self.gsv_height * self.clip_up
            # plt.show()
        print(f"All roots: {roots_all}")
        return roots_all, widths, DBH_points


    def getRoots(self, prominence=10, width=5, distance=20, plateau_size=(0, 150), prom_ratio=0.25):
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
            fig = plt.figure(figsize=(16, 8))
            ax_gsv = fig.add_subplot()
            ax_seg = fig.add_subplot()
            # plt.imshow(self.img_gsv_cv2)
            plt.imshow(ax_gsv)
            plt.show()

        roots_all = []
        widths = []
        # self.contoursNONE = self.contoursNONE[2:]

        verifieds_x = []
        verifieds_y = []

        try:

            if len(self.contoursNONE) < 1:  # no tree is found
                return roots_all, widths
        except:
            return roots_all, widths

        for cont_num, cont in enumerate(self.contoursNONE):  # for each contour from openCV2
            try:
                if len(cont) < 40:  # ignore small contour (i.g. polygon)
                    continue

                roots = []
                print("Processing contours #:", cont_num)

                # find peaks (i.g. roots).
                # peaks_idx, dic = scipy.signal.find_peaks(cont[:, 1], prominence=prominence, width=width,
                #                                          distance=distance,
                #                                          plateau_size=plateau_size)

                # print("cont:", cont)
                # peaks_idx, dic = scipy.signal.find_peaks(cont[:, 1], prominence=10, width=10, distance=20,
                #                                          plateau_size=10)

                simplified_con = self.simplify_contour(cont)
                


                peaks_idx, dic = scipy.signal.find_peaks(simplified_con, prominence=prominence, width=width,
                                                         distance=distance,
                                                         plateau_size=plateau_size)

                # if len(peaks_idx) > 0:
                #     print(peaks_idx)
                # else:
                #     print("Not peaks idx.")

                # peaks = cont[peaks_idx]  # get the col / row of peak

                peaks = np.concatenate((peaks_idx.reshape(-1, 1), simplified_con[peaks_idx].reshape(-1, 1)), axis=1) # get the col / row of peak

                ax.plot(peaks, 'o')

                ax.plot(simplified_con, 'x')
                # plt.show()

                verifieds_idx = []

                for i, peak in enumerate(peaks):
                    col = peak[0]
                    row = peak[1]

                    if self.conv_verify(row, col):  # check whether the peak fits to requirements
                        verifieds_x.append(col)
                        verifieds_y.append(row + self.gsv_height * self.clip_up)
                        verifieds_idx.append(i)
                        # print("Verified!", col, row)

                peaks = peaks[verifieds_idx]  # store the roots only
                roots.append(peaks)
                roots = np.concatenate(roots)
                # roots_all.append(roots)
                # print('\n', dic)

                if len(peaks_idx) == 0:  # if no roots
                    continue

                prominences = dic['prominences']  # height of peak (i.g. root)

                DBH_row = (roots[:, 1] - prominences[verifieds_idx] * prom_ratio).astype(
                    int)  # measure dimater in these rows
                # DBH_row = (roots[:, 1] - prominence).astype(int)

                # peaks_widths = dic['peaks_widths']
                # peaks_prominences = dic['prominences']

                if isDraw:
                    ax.hlines(y=dic["width_heights"] + self.gsv_height * self.clip_up, xmin=dic["left_ips"],
                              xmax=dic["right_ips"], color="C1")

                    ax.scatter(peaks, simplified_con[peaks], color='red', s=50)
                # plt.show()

                for idx, r in enumerate(DBH_row):  # measure each trunk

                    # print('\nr:', r)

                    line_x = []
                    line_y = []

                    DBH_idx = np.argwhere(simplified_con == r)  # get cols.
                    DBH_x = DBH_idx
                    # t = [x[0] for x in DBH_idx]  # get cols.

                    # DBH_x = simplified_con[DBH_idx]  # get cols.
                    #
                    # # remove the adjacent pixels
                    # temp = [DBH_x[0]]  # the most left col
                    # for x in range(1, len(DBH_x)):  # find the right col
                    #     if (DBH_x[x] - DBH_x[x - 1]) > 1:
                    #         temp.append(DBH_x[x])
                    # DBH_x = temp

                    # print('DBH_x: ', DBH_x)
                    #         print('width: ', abs(DBH_x[1] - DBH_x[0]))
                    # w = math.tan(math.radians(40)) * prominences[idx] * prom_ratio
                    #         print("w:", w)

                    MAX_DBH = 180  # maximum DBH, pixels
                    # if abs(roots[idx, 0] - DBH_x[0]) < (MAX_DBH / 2):   # remove the left edge of trunk is more than 80 pixels, ignore it.
                    if (DBH_x[-1] - DBH_x[0]) < MAX_DBH:  # remove the left edge of trunk is more than 80 pixels, ignore it.
                        roots_all.append(peaks[idx])
                        widths.append(DBH_x[1] - DBH_x[0])
                        line_x.append(DBH_x[1])
                        line_x.append(DBH_x[0])
                        line_y.append(r + self.gsv_height * self.clip_up)
                        line_y.append(r + self.gsv_height * self.clip_up)

                        if isDraw:
                            ax.add_line(Line2D(line_x, line_y, color='r'))
            except Exception as e:
                print("Error in getRoots():", e)
                continue

            if isDraw:
                ax.scatter(verifieds_x, verifieds_y, color='red', s=20)

                plt.savefig(os.path.join(r'J:\Research\Trees\west_trees_detected', \
                                         os.path.basename(self.seg_file_path).replace(".png", ".png")))
                # plt.show()

            if len(roots_all) > 0:
                roots_all = np.concatenate(roots_all).reshape((len(widths), -1))
                # roots_all[:, 1] = roots_all[:, 1] + self.gsv_height * self.clip_up

                if isDraw:   # draw results
                    ax.scatter(roots_all[:, 0], roots_all[:, 1], color='red', s=12)
                    plt.show()

            return roots_all, widths

def getRoors_mp(Prcesses_cnt = 5):
    # gsv = GSV_depthmap()
    # gpano = GPano()

    # global SEG_FOLDER  # = r'J:\Research\Trees\west_trees_seg\*.png'
    # global GSV_FOLDER  # = r'J:\Research\Trees\west_trees'
    # global COLOR_FOLDER #= r'J:\Research\Trees\west_trees_color'
    # global SAVED_FOLDER# = r'J:\Research\Trees\west_trees_detected'

    folder = SEG_FOLDER
    files = glob.glob(folder)[0:]
    # files = [r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Trees\datasets\Ottawa\tree_seg\7ad_9O5myatNpEMeyfhS0w_-75.765702_45.283713_0_135.41.png']

    files_mp = mp.Manager().list()
    for f in files:
        files_mp.append(f)

    pool = mp.Pool(processes=Prcesses_cnt)

    # print(pool)
    # print("files_mp:", files_mp[0])
    write = open(SAVED_FILE, 'w')
    write.writelines('X,Y,H,D,F\n')
    write.close()

    for i in range(Prcesses_cnt):
        pool.apply_async(test_getRoots, args=(files_mp,), callback=callback_write)
    pool.close()
    pool.join()


def callback_write(lines):
    with open(SAVED_FILE, 'a+') as f:
        f.writelines(str(lines))

def test_getRoots(files):
    # print("files: ", files)
    # img_file0 = r'56666_-75.135652_39.978263_20_102'
    # img_file0 = r'56615_-75.135871_39.977264_20_148'
    # img_file0 = r'56507_-75.142394_40.007453_20_134'
    # img_file0 = r'56436_-75.139345_40.019191_20_223'
    # img_file0 = r'56431_-75.139869_40.016782_20_305'
    # img_file0 = r'56389_-75.134487_39.993348_20_227'
    # img_file0 = r'56323_-75.138984_40.010789_20_237'
    # img_file0 = r'56279_-75.137075_39.981078_20_79'
    # img_file0 = r'56781_-75.116041_40.027007_20_353'  # failed
    # img_file0 = r'55019_-75.074928_40.028516_20_27'  # failed
    # img_file0 = r'56231_-75.134955_40.019275_20_319'
    # img_file0 = r'56072_-75.135019_39.975304_20_211'
    # img_file0 = '55953_-75.138752_40.021529_20_126'
    # img_file0 = '55960_-75.139287_40.021254_20_204'
    # img_file0 = '55926_-75.13393_40.023424_20_190'
    # img_file0 = r'55931_-75.134294_40.023345_20_128'
    # img_file0 = r'55835_-75.11125_40.027203_20_141'

    gsv = GSV_depthmap()
    gpano = GPano()


    # img_file = f'K:\\OneDrive_NJIT\\OneDrive - NJIT\\Research\\Trees\\datasets\\Philly\\Segmented_PSP\\{img_file0}.png'

    # folder = r'J:\Research\Trees\west_trees_seg\*.png'
    folder = SEG_FOLDER

    # files = glob.glob(folder)[4567:]

    results_lines = ''

    # writer = open(r'J:\Research\Trees\trees_only.txt', 'a')
    # writer.writelines("X,Y,H,D,F\n")

    while len(files) > 0:
    # for file in files:
        file = files.pop(0)

        print("Processing: ", file)

        try:
            tree_detect = tree_detection(seg_file_path=file)
            # roots_all, widths = tree_detect.getRoots0()
            roots_all, widths, DBH_points = tree_detect.getRoots_simplified()

            basename = os.path.basename(file)
            basename = basename.replace(".png", '')
            params = basename[:].split('_')

            thumb_panoId = '_'.join(params[:(len(params) - 4)])

            pano_lon = float(params[-4])
            pano_lat = float(params[-3])

            # thumb_panoId, _, _ = gpano.getPanoIDfrmLonlat(pano_lon, pano_lat)
            obj_json = gpano.getJsonfrmPanoID(thumb_panoId, dm=1)

            pano_heading = obj_json["Projection"]['pano_yaw_deg']
            pano_heading = math.radians(float(pano_heading))
            pano_pitch = obj_json["Projection"]['tilt_pitch_deg']
            pano_pitch = math.radians(float(pano_pitch))

            print("params:", params)
            # thumb_panoId = '_'.join(params[:(len(params) - 4)])


            thumb_heading = math.radians(float(params[-1]))

            thumb_theta0 = math.radians(float(params[-2]))

            thumb_phi0 = thumb_heading - pano_heading

            obj_json = gpano.getJsonfrmPanoID(thumb_panoId, dm=1)

            pano_tilt_yaw = obj_json["Projection"]['tilt_yaw_deg']
            pano_tilt_yaw = math.radians(float(pano_tilt_yaw))

            fov = math.radians(90)
            h = 768
            w = 1024

            fov_h = fov
            fov_v = atan((h * tan((fov_h / 2)) / w)) * 2

            pixel_idx = np.argwhere(np.array(Image.open(file)) > -1)  #


            sphs = gsv.castesian_to_shperical0(thumb_theta0, \
                                                thumb_phi0, pano_pitch, \
                                                pano_tilt_yaw, fov_h, h, w)
            sph_phi = sphs[pixel_idx[:, 0], pixel_idx[:, 1], 1]
            sph_theta = sphs[pixel_idx[:, 0], pixel_idx[:, 1], 0]

            # plt_x = [math.degrees(x) for x in sidewalk_sph_phi]
            # plt_y = [math.degrees(x) for x in sidewalk_sph_theta]
            # plt.scatter(sidewalk_sph_phi, sidewalk_sph_theta)
            # plt.show()

            sph = np.stack((sph_phi, sph_theta), axis=1)
            # print('len of sidewalk_sph :', len(sidewalk_sph))
            # print('sidewalk_sph[0]:', sidewalk_sph[0])
            dm = gsv.getDepthmapfrmJson(obj_json)

            pano_H = obj_json["Location"]['elevation_wgs84_m']

            pointcloud = gsv.getPointCloud2(sph, thumb_heading, thumb_theta0, dm,
                                             pano_lon, pano_lat, pano_H, pano_heading, pano_pitch)

            print(pointcloud.shape)

            # print(pointcloud[-10:])

            print(roots_all, widths)
            if len(roots_all) > 0:
                roots_xyz = pointcloud[roots_all[:, 1] * w + roots_all[:, 0] -1]
                roots_xyz = roots_xyz[roots_xyz[:,  3] < 20 * 3]
                for t in roots_xyz:

                    # writer.writelines(f"{t[0]},{t[1]},{t[2]},{t[3]},{os.path.basename(file)}\n")
                    results_lines += (f"{t[0]},{t[1]},{t[2]},{t[3]},{os.path.basename(file)}\n")
                    print("Trees: ", f"{t[0]}, {t[1]}, {t[2]}, {t[3]}")

            # writer.flush()
            # writer.flush()
            # typically the above line would do. however this is used to ensure that the file is written
            # os.fsync(writer.fileno())



    # plt.imshow(tree_detect.opened)
    # plt.scatter(roots_all[:, 0], roots_all[:, 1])
    # plt.show()
        except Exception as e:
            print("Error: ", e, file)
    # writer.close()
    return results_lines

if __name__ == "__main__":

    # global SEG_FOLDER
    # global GSV_FOLDER
    # global COLOR_FOLDER
    # global SAVED_FOLDER


    # test_getRoots()
    getRoors_mp(Prcesses_cnt=1)