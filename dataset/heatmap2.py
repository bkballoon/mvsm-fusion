import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans


# 注视点数据要求有duration属性，这里添加上数据的注释时长属性
def process_gaze_data(txt_path):
    gaze_point_list = []

    file = open(txt_path, 'r')
    all_lines = file.readlines()
    gaze_ball_position_lists = []
    # 将字符串数值化
    for line in all_lines:
        numbers = line.split(',')
        gaze_ball_position = np.array([numbers[0], numbers[1], numbers[2]], np.float32)
        gaze_ball_position_lists.append(gaze_ball_position)
    gaze_ball_position_lists = np.array(gaze_ball_position_lists)

    # 将unity坐标系转换成opencv坐标系
    data_lines = int(len(gaze_ball_position_lists) / 3)
    for k in range(data_lines * 0, data_lines * 2):
        # 坐标系挪到左上角
        gaze_ball_position_lists[k] += np.array([0.5, -0.5, 0])
        # y轴翻转得到opencv坐标系
        gaze_ball_position_lists[k][1] *= -1

    # 将观察者的数据显示到图像上
    dist = 1 / 512

    for k in range(data_lines * 0, data_lines * 2):
        gaze_x = int(gaze_ball_position_lists[k][0] / dist)
        gaze_y = int(gaze_ball_position_lists[k][1] / dist)
        gaze_y += 18 # 将偏上的点挪下来
        if 0 <= gaze_x < 512 and 0 <= gaze_y < 512:
            gaze_point_list.append((gaze_x, gaze_y))

    # gaze_point_list = np.array(gaze_point_list)

    return gaze_point_list

def gaussian(sizex, sizey, sigma=33, center=None, fix=1):

    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))

    return fix * np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def fixation_density_map(fixation, W, H, imgfile, alpha=0.5, threshold=10):
    """
    :param fixation:(x,y,duration)
    :param width:widht of image
    :param height:height of image
    :param imgfile:source image
    :param alpha:comparency of the mask
    :param threshold:
    :return:
    """
    heatmap = np.zeros((H, W), np.float32)
    # 进度条显示
    for n_subject in tqdm(range(fixation.shape[0])):
        heatmap += gaussian(W, H, 33, (fixation[n_subject, 0], fixation[n_subject, 1]),
                            fixation[n_subject, 2])

    heatmap = heatmap / np.amax(heatmap)
    heatmap = heatmap * 255
    heatmap = heatmap.astype("uint8") # 格式转换

    # gray color
    # return heatmap

    # imgfile2 = np.ones(W*H*3, dtype=np.uint8).reshape([W, H, 3])*255
    # imgfile = imgfile2

    # cv2.imwrite("/home/hmao/my/lab/textured_mesh_saliency/Dataset/pic/paper_feline/gd_truth_front_feline_all.png", heatmap)
    # exit()

    if imgfile.any():
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap_color = heatmap
        # heatmap_color = np.repeat(heatmap_color, 3, axis=2)

        # Create mask
        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2) # (w,h,1)->(w,h,3)

        # Put the mask(heatmap) onto the image, which means merging the two image
        merge = imgfile * mask + heatmap_color * (1 - mask)
        merge = merge.astype("uint8")
        merge = cv2.addWeighted(imgfile, 1 - alpha, merge, alpha, 0)
        return merge

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

def norm(p):
    length = len(p)
    end = 0
    for i in range(length):
        end += p[i] * p[i]
    return np.sqrt(end)


def unique(gaze_point_list):
    end_gaze_points = []
    pre_uni = norm(gaze_point_list[0])
    uni_count = 1
    for gaze_point_index in range(1, len(gaze_point_list)):
        gaze_point = gaze_point_list[gaze_point_index]
        cur_uni = norm(gaze_point)
        if cur_uni == pre_uni:
            uni_count += 1
        else:
            end_gaze_points.append([gaze_point[0], gaze_point[1], uni_count])
            pre_uni = cur_uni
            uni_count = 1

    end_gaze_points = np.array(end_gaze_points)

    return end_gaze_points


if __name__ =="__main__":
    for p in range(8):
        gaze_points = []
        for i in ["zy", "zy2", "ybxz", "sxy", "lxj", "llxj", "hlg"]:
            txt_path = "D:/Learn/Dataset/pic/fixation/" + i + '/' + str(p) + '.txt'
            pic_path = "D:/Learn/Dataset/pic/fixation/all/" + str(p) + '.png'

            # Load image file
            img = cv2.imread(pic_path)
            img = cv2.resize(img, (512, 512))

            # Generate toy fixation data
            num_subjects = 20
            H, W, _ = img.shape
            gaze_points += process_gaze_data(txt_path)

        # initial it with numpy
        gaze_points = np.array(gaze_points, dtype=np.int32)

        X = gaze_points[:, 0:2]
        cluster_n = 14
        # 初始化14个原始聚类中心
        y_pred = KMeans(n_clusters=cluster_n, random_state=9).fit_predict(X)
        print("a, b = {}, {}".format(len(gaze_points), len(y_pred)))

        cluster_sum = np.zeros(cluster_n)
        for pred in y_pred:
            cluster_sum[pred] += 1
        cluster_sum_cp = [i for i in cluster_sum]
        times = sorted(cluster_sum_cp)[int(cluster_n/4)]
        delete_index = []
        for cs_i in range(len(cluster_sum)):
            cs = cluster_sum[cs_i]
            if cs <= times:
                delete_index.append(cs_i)

        heatmap3 = np.zeros(512 * 512).reshape([512, 512])
        gaze_points3 = []
        for i in range(len(gaze_points)):
            line = gaze_points[i]
            x, y = line[0], line[1]
            if y_pred[i] not in delete_index:
                heatmap3[y][x] = int(y_pred[i])
                gaze_points3.append([x, y])

        gaze_points1 = unique(gaze_points)
        gaze_points3 = unique(gaze_points3)

        print(gaze_points3.shape)
        print(gaze_points3[0].shape)

        # mother_path = "D:\\Learn\\lab\\mvsm\\Dataset\\distribute\\"
        # path = mother_path + str(p) + ".txt"
        # file = open(path, "w")
        # lines = gaze_points3[:, 2]
        # lines = [str(i)+'\n' for i in lines]
        # file.writelines(lines)
        # file.close()

        # heatmap1 = fixation_density_map(gaze_points1, W, H, img)
        # heatmap3 = fixation_density_map(gaze_points3, W, H, img)

        # cv2.imwrite("/home/hmao/my/lab/textured_mesh_saliency/Dataset/pic/paper_experiment/fig/" + str(p) + "_3.png", heatmap1)
        # cv2.imwrite("/home/hmao/my/lab/textured_mesh_saliency/Dataset/pic/paper_experiment/fig/" + str(p) + "_2.png", heatmap3)
        # cv2.imwrite(str(p) + "_3.png", heatmap1)

        # plt.subplot(121)
        # plt.imshow(heatmap1)
        # plt.subplot(122)
        # plt.imshow(heatmap3)
        # plt.show()