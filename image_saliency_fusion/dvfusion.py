import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

# from skimage import color
# from py_2dto3d.pqft import PQFTLib

from mayavi import mlab
import myutil

from saliency_method import *
# from gbvs import gbvs_saliency
from skimage import exposure

def get_fmc(mc, vertices, faces):
    fmc = [0 for i in range(len(faces))]
    for face_i in range(len(faces)):
        face_3p = faces[face_i]
        p1, p2, p3 = face_3p[0], face_3p[1], face_3p[2]
        w1, w2, w3 = myutil.barycenter(vertices[p1], vertices[p2], vertices[p3])
        fmc[face_i] = w1 * mc[p1] + w2 * mc[p2] + w3 * mc[p3]
    return fmc

def test_rgb_to_index_function():
    # put the one image saliency to the model
    vertices, faces = load_obj("../data/bunny.obj")
    face_saliency = []
    for i in range(len(faces)):
        face_saliency.append(0)
        # face_saliency.append(i / len(faces))

    color_img = cv2.imread("../video/6.png")
    index_img = cv2.imread("../video1/6.png")
    # color_img = cv2.imread("../out/+x0.png")
    # index_img = cv2.imread("../out/+x1.png")


    index_list = []
    color_lab = ft_saliency(color_img)

    faces_len = len(faces)
    size = color_lab.shape
    for i in range(size[0]):
        for j in range(size[1]):
            saliency = color_lab[i][j]
            index = rgb_to_face(index_img[i][j]) - 1
            if 0 < index < faces_len:
                index_list.append(index)
                face_saliency[index] = saliency

    # for i in index_list:
    #     face_saliency[i] = random.random()

    print("index_list = {}".format(len(index_list)))
    print("no repeat ", len(list(set(index_list))))

    myutil.mayavi_with_custom_face(vertices, faces, face_saliency)

def how_to_convert_rgb_to_index():
    index_img = cv2.imread("../video1/6.png")
    size = index_img.shape
    vertices, faces = load_obj("../data/bunny.obj")
    faces_len = len(faces)
    print(index_img[383][485])

    # for i in range(size[0]):
    #     for j in range(size[1]):
    #         pixel = index_img[i][j]
    #         # if np.sum(pixel) != 0 and np.sum(pixel) != 255 * 3 and pixel[0] < 100:
    #         #     print(index_img[i][j])
    #         if pixel[0] == pixel[1] == pixel[2]:
    #             index_img[i][j] = np.array([0, 0, 0], np.uint8)
    #         else:
    #             print(pixel)

    cv2.imshow("jkfls", index_img)
    cv2.waitKey(0)

def show_spectral_model():
    path = "../data/vertex_s.txt"
    saliency = open(path).readlines()[0].split(' ')
    saliency = np.array(saliency, np.float)
    vertices, faces, nnorms = load_obj("../data/bunny_simp_30_0.obj")
    vertex_saliency = []
    for i in range(len(faces)):
        p1, p2, p3 = faces[i]
        vertex_saliency.append((saliency[p1] + saliency[p2] + saliency[p3]) / 3)
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
    myutil.mayavi_with_custom_point(vertices, faces, vertex_saliency)
    mlab.show()

def load_obj(filename):
    vertices_ = []
    faces_ = []
    normals_ = []
    for line in open(filename):
        if len(line) == 0:
            continue
        if line.startswith('#'):
            continue
        values = line.split(' ')
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            v = [v[0], v[1], v[2]]
            vertices_.append(v)
        elif values[0] == 'f':
            f = [int(values[1].split('/')[0])-1,
                 int(values[2].split('/')[0])-1,
                 int(values[3].split('/')[0])-1]
            faces_.append(f)
        elif values[0] == 'vn':
            n = [float(x) for x in values[1:4]]
            n = [n[0], n[1], n[2]]
            normals_.append(n)
    vertices_ = np.array(vertices_)
    faces_ = np.array(faces_)
    normals_ = np.array(normals_)
    return [vertices_, faces_, normals_]


def read_image(path):
    after_png = cv2.imread(path)
    return after_png

def rgb_to_face(rgb):
    k = 255 * 255 * rgb[0] + 255 * rgb[1] + 1 * rgb[2]
    return k

def face_to_rgb(face):
    rgb = []
    c = face
    for i in range(3):
        t = c % 255
        c = int(c / 255)
        rgb.append(t)
    rgb.reverse()
    return rgb

def f2v_mc(vertices, faces, face_saliency, fmc):
    v_to_triangles = myutil.get_vertex_to_face(faces)
    # face saliency to vertex saliency based on the area of triangle
    vertex_saliency = np.zeros(len(vertices))
    for i in range(len(vertices)):
        tris = v_to_triangles[i]
        local_mcs = []
        for tri_index in tris:
            local_mcs.append(fmc[tri_index])
            vertex_saliency[i] += face_saliency[tri_index] * fmc[tri_index]
        vertex_saliency[i] = vertex_saliency[i] / (max(local_mcs) - min(local_mcs))
        vertex_saliency[i] = vertex_saliency[i] / len(tris)
    return vertex_saliency


def write_vs(data_name, vertex_saliency):
    path = data_name + '_vs_salicon.txt'
    file = open(path, 'a')
    for i in vertex_saliency:
        file.write(str(i))
        file.write('\n')

# 获取每个triangle的平均下的imageSaliency
def generate_multiview_saliency(data_name, number):
    # load the OBJ file
    base_path_win = "E:\\Dataset\\textured_model\\"
    base_path_linux = "/home/hmao/my/lab/textured_mesh_saliency/Dataset/textured_model/"
    vertices, faces, nnorms = load_obj(
        base_path_win + \
        data_name + "\\" + data_name + ".obj")

    print("v is {}, f is {}".format(len(vertices), len(faces)))

    # load the multi-view images
    rgb_pic_name = []
    dis = int(360 / number)
    for i in range(number):
        path = "E:\\Dataset\\pic\\paper_" + \
                data_name + "\\video0\\" + str(dis * i + 15) + ".png"
        rgb_pic_name.append(path)
    print(rgb_pic_name)
    # load the multi-view mapping images with face index
    index_color_pic_name = []
    for i in range(number):
        path = "E:\\Dataset\\pic\\paper_" + \
                data_name + "\\video1\\" + str(dis * i + 15) + ".png"
        index_color_pic_name.append(path)

    # get the pic shape
    path1 = rgb_pic_name[0]
    img_shape = cv2.imread(path1).shape
    # container which save the imgs or lab saliency
    imgs = []
    labs = []
    index_img = []
    # list the images
    print("pre processing--> pull in the labs stack")
    for i in range(len(rgb_pic_name)):
        imgs.append(cv2.imread(rgb_pic_name[i]))
        index_img.append(cv2.imread(index_color_pic_name[i]))
        labs.append(lc_saliency(cv2.imread(rgb_pic_name[i])))
        # labs.append(ac_saliency(cv2.imread(rgb_pic_name[i]), 100, 400, 3))
        # labs.append(ft_saliency(cv2.imread(rgb_pic_name[i])))
        # labs.append(gbvs_saliency(cv2.imread(rgb_pic_name[i])))
        # labs.append(salicon_saliency(data_name, i, dis))
        # labs.append(deep_saliency(data_name, i, dis))

    # show the labs
    # for i in range(len(labs)):
    #     lab = labs[i]
    #     plt.subplot(2, len(labs) / 2, i + 1)
    #     plt.imshow(lab)
    # plt.show()
    # exit()

    print("--> next is gaussian filter")
    # gaussian filter
    for i in range(len(labs)):
        lab = labs[i]
        lab = cv2.GaussianBlur(lab, (3, 3), 0.8)
        labs[i] = lab

    # traverse all the pixel to mark all the color index and know the face saliency
    res_dict_list = []
    for p in range(len(imgs)):
        print(">> current do 2D to 3D  {} img".format(p))
        pic_dict = dict()  # each pic_dict is {face_index:face_saliency}
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                # for all the picture and its pixels
                # compute the color to index and save it
                pixel = index_img[p][i][j]
                face_index = rgb_to_face(pixel) - 1
                if face_index not in pic_dict.keys():
                    pic_dict[face_index] = [labs[p][i][j]]
                else:
                    pic_dict[face_index].append(labs[p][i][j])

        for key in pic_dict.keys():
            pic_dict[key] = np.mean(pic_dict[key])  # face_index->multi saliency, so mean them

        res_dict_list.append(pic_dict)
    # get the res_dict->[img1{face_index:face_saliency_value}, ...]
    return vertices, faces, res_dict_list

def unified_scale_song(vertices, faces, res_dict_list):
    v_to_face = myutil.get_vertex_to_face(faces)
    v_n_dist = myutil.get_v_normal_dist(v_to_face, vertices, faces)

    vertex_saliency = np.zeros(len(vertices))
    # current view face_saliency
    for res_dict in res_dict_list:
        ver_s = [[0] for i in range(len(vertices))]
        for k, v in res_dict.items():
            if k < len(faces):
                three_p = faces[k]
                ver_s[three_p[0]].append(v)
                ver_s[three_p[1]].append(v)
                ver_s[three_p[2]].append(v)
        for vs_i in range(len(vertices)):
            if len(ver_s[vs_i]) >= 3:
                ver_s[vs_i] = np.mean(ver_s[vs_i])
            else:
                ver_s[vs_i] = 0
        for vsi in range(len(ver_s)):
            vs = ver_s[vsi]
            if vs > 0:
                ver_s[vsi] = np.exp(1-v_n_dist[vsi])/np.exp(1-vs)
            else:
                vs = 0
        ver_s = np.array(ver_s)
        vertex_saliency = vertex_saliency + ver_s

    return vertex_saliency / len(res_dict_list)

def unified_scale_song_edit(vertices, faces, res_dict_list):
    tri_to_tris = myutil.compute_tri_with_tri(vertices, faces)
    # container
    tri_to_n_dist = myutil.get_tri_normal_dist(tri_to_tris, vertices, faces)
    face_saliency = np.zeros(len(faces))
    for res_dict in res_dict_list:
        fs = np.zeros(len(faces))
        for k, v in res_dict.items():
            if k < len(faces):
                fs[k] = v
        for fs_i in range(len(fs)):
            fs[fs_i] = np.exp(1-tri_to_n_dist[fs_i])/np.exp(1-fs[fs_i])

        fs = np.array(fs)
        face_saliency = face_saliency + fs
    face_saliency = face_saliency / len(res_dict_list)
    vertex_saliency = myutil.f2v(vertices, faces, face_saliency)

    return vertex_saliency


# input base1 pic_dict1 and base2 pic_dict2, output base1 pic_dict1 and pic_dict2
def unified_scale_median(number, mc, fmc, vertices, faces, res_dict_list):
    tri_to_tris = myutil.compute_tri_with_tri(vertices, faces)
    tri_to_n_dist = myutil.get_tri_normal_dist(tri_to_tris, vertices, faces)

    high_curvature = set()
    for num_i in range(number - 1):
        print(">> current fusion index is {}".format(num_i))
        res_dict = [res_dict_list[num_i], res_dict_list[num_i + 1]]

        # face_saliency is the model_saliency_map
        face_saliency = []
        for i in range(len(faces)):
            face_saliency.append([0])

        for d in res_dict:
            for k, v in d.items():
                if k < len(faces):
                    face_saliency[k].append(v)

        # f_mc_list = [mc1, mc2, mc3...], f_mc_list_index = [mc1->face_index, ... ,]
        f_mc_list = []
        f_mc_list_index = []
        for index in range(len(face_saliency)):
            f = face_saliency[index]
            three_p = faces[index]
            f_mc = (mc[three_p[0]] + mc[three_p[1]] + mc[three_p[2]]) / 3
            if len(f) > 2:
                f_mc_list.append(f_mc)
                f_mc_list_index.append(index)

        f_mc_peak = sorted(f_mc_list)[int(len(f_mc_list) / 2)]
        f_mc_peak_index = f_mc_list_index[f_mc_list.index(f_mc_peak)]
        final_rate = res_dict[1][f_mc_peak_index] / res_dict[0][f_mc_peak_index]

        for i in range(len(f_mc_list_index)):
            index = f_mc_list_index[i]
            f_mc = fmc[index]
            face_fmc[index] = f_mc

        for f_mc_peak in f_mc_list:
            f_mc_peak_index = f_mc_list_index[f_mc_list.index(f_mc_peak)]
            print(f_mc_peak_index)
            if res_dict[0][f_mc_peak_index] > 0:
                rate = res_dict[1][f_mc_peak_index] / res_dict[0][f_mc_peak_index]
                rate_list.append(rate)
                face_rate[f_mc_peak_index] += rate
                c1.append(res_dict[1][f_mc_peak_index])
                c2.append(res_dict[0][f_mc_peak_index])
        print(rate_list[:10])
        
        mean_rate = np.sum(rate_list) / len(rate_list)
        median_rate = np.median(rate_list)
        print("mean_rate = ", mean_rate)
        print("median_rate = ", median_rate)

        final_rate = median_rate
        for k, v in res_dict[1].items():
            res_dict[1][k] /= final_rate

        res_dict_list[num_i] = res_dict[0]
        res_dict_list[num_i+1] = res_dict[1]

    # create a face_saliency container
    face_saliency = []
    for i in range(len(faces)):
        face_saliency.append([0])

    for d in res_dict_list:
        for k, v in d.items():
            if k < len(faces):
                face_saliency[k].append(v)

    for f_s_i in range(len(face_saliency)):
        f_s = face_saliency[f_s_i]
        face_saliency[f_s_i] = np.mean(f_s)

    # high_curvature = list(high_curvature)
    # for face_i in high_curvature:
    #     # augment
    #     augment = np.exp(1 - tri_to_n_dist[face_i]) / np.exp(1 - face_saliency[face_i])
    #     augment = 1.0
    #     face_saliency[face_i] *= augment

    # 基于面积的加权；基于曲率的加权；多尺度加权；假设点的SaliencyMap=面的Saliency的平均
    vertex_saliency = myutil.f2v(vertices, faces, face_saliency)
    # vertex_saliency = f2v_mc(vertices, faces, face_saliency, fmc)
    # vertex_saliency = f2v_mythod(vertices, faces, face_saliency, fmc)

    return vertex_saliency

def my_method(data_name, number):
    # get the multi-view images' saliency
    # res_dict_list = [{face_index:face_saliency_mean}, ... ,]
    vertices, faces, res_dict_list = generate_multiview_saliency(data_name, number)
    # get the mean curvature of the model
    mc_path = "E:\\Dataset\\textured_model\\" + \
               data_name + "\\saliency.txt"

    file = open(mc_path)
    lines = file.readlines()
    mc = np.array([np.float(s) for s in lines])
    mc = np.array([1.0 for s in lines])
    mc = exposure.equalize_hist(mc)
    fmc = get_fmc(mc, vertices, faces)

    # recurrent two images move on with median rate in intersection part
    time1 = time.time()
    vertex_saliency = unified_scale_median(number, mc, fmc, vertices, faces, res_dict_list)
    time2 = time.time()
    print("时间是", time2 - time1)

    # mayavi figure
    white = (1, 1, 1)
    black = (0, 0, 0)
    fig = mlab.figure(size=(512, 512 + 48), bgcolor=white)
    # myutil.mayavi_with_custom_face(vertices, faces, face_saliency)
    myutil.mayavi_with_custom_point(vertices, faces, vertex_saliency)
    mlab.show()


def display_labs(labs):
    lab_len = len(labs)
    for i in range(len(labs)):
        index = i+1
        plt.subplot(2, int(lab_len/2), index)
        plt.imshow(labs[i], cmap='gray')
    plt.show()
    return

def show_bunny():
    vertices, faces, nnorms = load_obj("/home/hmao/my/lab/textured_mesh_saliency/Dataset/textured_model/bunny/bunny.obj")
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces,
                         color=(0.6, 0.6, 0.6))
    mlab.show()


if __name__ == "__main__":
    # how_to_convert_rgb_to_index()
    # test_rgb_to_index_function()
    # mean_image_saliency_model(number=2)
    # show_spectral_model()
    # show_bunny()
    # myutil.compute_point_normal()

    my_method("bunny", 4)
    # data_name = "bunny"
    # vertices, faces, nnorms = load_obj("/home/hmao/my/lab/textured_mesh_saliency/Dataset/textured_model/" + data_name + "/" +
    #                                    data_name + ".obj")
    # tri_to_tris = myutil.compute_tri_with_tri(vertices, faces)
    # tri_to_n_dist = myutil.get_tri_normal_dist(tri_to_tris, vertices, faces)

