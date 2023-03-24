import numpy as np
import cv2
import copy

# FT saliency detection
# frequency-tuned salient region detection
def ft_saliency(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l_mean = np.mean(gray_lab[:, :, 0])
    a_mean = np.mean(gray_lab[:, :, 1])
    b_mean = np.mean(gray_lab[:, :, 2])
    lab = np.square(gray_lab - np.array([l_mean, a_mean, b_mean]))
    lab = np.sum(lab, axis=2)
    lab = lab / np.max(lab)
    # lab = 1 - lab # inverse the lab, cus the inverse-lab match the real situation

    return lab  # 这不是标准的lab空间的图片

# LC saliency detection
# visual attention detection in video sequences using spatiotemporal cues
def cal_dist(hist):
    dist = {}
    for gray in range(256):
        value = 0.0
        for k in range(256):
            value += hist[k][0] * abs(gray - k)
        dist[gray] = value
    return dist
def lc_saliency(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_height = image_gray.shape[0]
    image_width = image_gray.shape[1]
    image_gray_copy = np.zeros((image_height, image_width))
    # 直方图，统计图像中每个灰度值的数量
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    gray_dist = cal_dist(hist_array)  # 灰度值与其他值的距离
    # print(gray_dist)
    for i in range(image_width):
        for j in range(image_height):
            temp = image_gray[j][i]
            image_gray_copy[j][i] = gray_dist[temp]
    image_gray_copy = (image_gray_copy - np.min(image_gray_copy)) / (np.max(image_gray_copy) - np.min(image_gray_copy))
    return image_gray_copy

# AC saliency, core idea is that saliency is determined as the local contrast of an image
# region with respect to its neighborhood at various scales.
# salient region detection and segmentation
def ac_saliency(src_img, minR2, maxR2, scale):
    lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2Lab)
    row, col, _ = src_img.shape
    Sal_org = np.zeros(row * col).reshape([row, col])
    Sal = np.zeros(row * col, dtype=np.uint8).reshape([row, col, 1])

    max_v = 0
    min_v = pow(10, 10)
    for k in range(scale):
        len = int((maxR2 - minR2) * k / (scale - 1) + minR2)
        filter = cv2.blur(lab, (len, len))
        for i in range(row):
            for j in range(col):
                p = lab[i][j]
                p = np.array(p, dtype=np.int)
                p1 = filter[i][j]
                val = np.sqrt(
                    (p[0] - p1[0]) * (p[0] - p1[0]) +
                    (p[1] - p1[1]) * (p[1] - p1[1]) +
                    (p[2] - p1[2]) * (p[2] - p1[2])
                )
                Sal_org[i][j] += val
                if k == scale - 1:
                    max_v = max(max_v, Sal_org[i][j])
                    min_v = min(min_v, Sal_org[i][j])

    for i in range(row):
        for j in range(col):
            Sal[i][j] = (Sal_org[i][j] - min_v) * 255 / (max_v - min_v)

    return Sal


# salicon method
def salicon_saliency(data_name, index, dis):
    salicon_path = "/home/simula/Pic/paper_" + data_name +"/salicon/"
    txt_path = str(index * dis + 15) + ".txt"
    file = open(salicon_path + txt_path, 'r')
    all_lines = file.readlines()
    img_mat = np.zeros(800 * 800).reshape([800, 800])
    for x in range(len(all_lines)):
        line = all_lines[x]
        ys = line.split(' ')
        for y in range(len(ys)):
            img_mat[x][y] = ys[y]
    return img_mat


# deep method
def deep_saliency(data_name, index, dis):
    deep_path = "/home/simula/Pic/paper_" + data_name +"/deep/"
    png_path = str(index * dis + 15) + ".png"
    img = cv2.imread(deep_path + png_path)
    img_mat = np.zeros(800*800).reshape([800, 800])
    for x in range(800):
        for y in range(800):
            pixel = img[x][y]
            img_mat[x][y] = pixel[0]/255
    return img_mat
# img_path = '/home/simula/Pic/paper/video0/15.png'
# img = cv2.imread(img_path)
# # sal = ac_saliency(img, 100, 400, 3)
# sal = lc_saliency(img)
# print(sal.shape)
# print(type(sal[0][0]))
# cv2.imshow("sal", sal)
# cv2.waitKey()

