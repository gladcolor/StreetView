from __future__ import print_function
import cv2
import numpy as np
from shapely.geometry import Polygon
# from centerline.geometry import Centerline
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

import GPano
#import GPano.GSV_depthmap
#%%

#%%
#gpano = GPanorama()

from label_centerlines import get_centerline


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h




if __name__ == '__main__':
    dm = GPano.GSV_depthmap()
    #
    # # Read reference image
    # refFilename = r"G:\My Drive\right.jpg"
    # print("Reading reference image : ", refFilename)
    # imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    #
    # # Read image to be aligned
    # imFilename = r"G:\My Drive\left.jpg"
    # print("Reading image to align : ", imFilename);
    # im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    #
    # print("Aligning images ...")
    # # Registered image will be resotred in imReg.
    # # The estimated homography will be stored in h.
    # imReg, h = alignImages(im, imReference)
    #
    # # Write aligned image to disk.
    # outFilename = "aligned.jpg"
    # print("Saving aligned image : ", outFilename);
    # cv2.imwrite(outFilename, imReg)
    #
    # # Print estimated homography
    # print("Estimated homography : \n", h)

    from shapely.geometry import Polygon
    from centerline.geometry import Centerline
    import cv2

    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io
    from PIL import Image

    img_file = r'I:\DVRPC\Fill_gap\StreetView\images\A6cVqLl94hZqpnEJkLir-g_-75.499092_40.401787_0_51.09_landcover.png'
    img_file = r'I:\DVRPC\Fill_gap\StreetView\images\sG51XsNz_X9EWGv_nU8jaw_-74.848704_40.150078_0_134.31_landcover.png'
    img_cv = cv2.imread(img_file)
    img_io = io.imread(img_file)
    img_pil = Image.open(img_file)
    # plt.imshow(img_io)
    # plt.show()

    np_img = np.array(img_pil)

    unique_elements, counts_elements = np.unique(np_img, return_counts=True)
    print(unique_elements)
    print(counts_elements)

    np_img = np.where((np_img == 12) | (np_img == 53), 255, 0).astype(np.uint8)
    unique_elements, counts_elements = np.unique(np_img, return_counts=True)
    print(unique_elements)
    print(counts_elements)

    pimg = Image.fromarray(np_img)

    # plt.imshow(pimg)
    # plt.show()

    # cannot convert png to cv2 format.
    pimg.save("temp.png")

    # img1 = Image.open("temp.png")
    # plt.imshow(img1)
    # plt.show()

    img_cv2 = cv2.imread("temp.png", cv2.IMREAD_UNCHANGED)

    cv2.imshow('landcover', cv2.imread(img_file, cv2.IMREAD_UNCHANGED))

    # plt.imshow(img_cv2)
    # plt.show()

    img_np = np.array(img_cv2)
    unique_elements, counts_elements = np.unique(img_np, return_counts=True)
    print(unique_elements)
    print(counts_elements)

    cv2.imshow('MORPH_CLOSE', cv2.imread("temp.png", cv2.IMREAD_UNCHANGED))

    ret,thresh = cv2.threshold(img_cv2, 0, 255,0)
    cv2.imshow('thresh', thresh)

    g = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))


    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, g)
    cv2.imshow('MORPH_CLOSE', closed)

    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, g)
    cv2.imshow('MORPH_OPEN', opened)


    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, g)
    cv2.imshow('MORPH_CLOSE2', closed)

  opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, g)
    cv2.imshow('MORPH_OPEN2', opened)

    backtorgb = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2RGB)
    # print(thresh[:, 100

    # plt.imshow(img_cv2)
    # plt.show()

    contours, hierarchy = cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[3]
    # print(contours[0])

    contour = np.squeeze(contours[0])


    # attributes = {"id": 1, "name": "polygon", "valid": True}
    # centerline1 = Centerline(polygon,interpolation_distance=10)


    backtorgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
    cv2.imshow("Original", backtorgb)
    t = cv2.drawContours((backtorgb), contours, -1, (0, 255, 0), 1)
    print(len(contours))

    lineThickness = 1

    # for line in centerline1:
    #     # print(list(line.coords)[0])
    #     coords = list(line.coords)
    #     start = (int(coords[0][0]), int(coords[0][1]))
    #     end   = (int(coords[1][0]), int(coords[1][1]))
    #     cv2.line(t, start, end, (0, 0, 255), lineThickness)

    for cont in contours:
        contour = np.squeeze(cont)
        polygon = Polygon(contour)
        print(polygon)
        centerline1 = get_centerline(polygon, segmentize_maxlen=8, max_points=3000, simplification=0.05, smooth_sigma=5)
        print(centerline1)
        line = centerline1
        # print(list(line.coords)[0])
        coords = list(line.coords)
        print(len(coords), coords)
        for idx, coord in enumerate(coords[:-1]):
            # print(idx, )
            start = (int(coords[idx][0]), int(coords[idx][1]))
            end   = (int(coords[idx + 1][0]), int(coords[idx + 1][1]))
            cv2.line(t, start, end, (0, 0, 255), lineThickness)

    cv2.imshow('closed',t)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()



