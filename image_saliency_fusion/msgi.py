import random

import sys 
sys.path.append("..\\..")
import cv2
from skimage import exposure
import numpy as np
from mayavi import mlab
import yang.myutil as myutil
from matplotlib import pyplot as plt

def rotation(img):
    rows, cols = img.shape[:2]
    M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    img = cv2.warpAffine(img, M2, (rows, cols))
    img = cv2.flip(img, 1)
    return img

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


def read_fs(path):
    txt = open(path)
    all_lines = txt.readlines()
    fs = [np.float(line) for line in all_lines]
    return fs

def augmentation(fs):
    return fs

def write_vs(path, vertex_saliency):
    file = open(path, 'w')
    for i in vertex_saliency:
        file.write(str(i))
        file.write('\n')

def mayavi_pcl(vertices, texels):
    n = len(vertices)  # number of points
    print(len(vertices))
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    x /= 100
    y /= 100
    z /= 100

    # x, y, z = np.random.random((3, n))
    # z = np.zeros(z.shape)
    # texels = [np.array([100, 200, 300, 255], dtype=np.uint8) for index in range(len(vertices))]
    rgba = np.random.randint(0, 256, size=(n, 4), dtype=np.uint8)
    rgba[:, -1] = 255  # no transparency
    pts = mlab.pipeline.scalar_scatter(x, y, z)  # plot the points
    pts.add_attribute(texels, 'colors')  # assign the colors to each point
    # pts.add_attribute(rgba, 'colors') # assign the colors to each point
    pts.data.point_data.set_active_scalars('colors')
    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = 0.1  # set scaling for all the points
    g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size

def compute_lbp(texels, faces, texels_gray, v_to_face):
    lbp = []
    for num in range(len(texels)):
        one_ring = myutil.get_one_ring_vertex(v_to_face, num, faces)
        one_ring.remove(num)
        lb = []
        for one in one_ring:
            if texels_gray[num] > texels_gray[one]:
                lb.append(1)
            else:
                lb.append(0)
        value = 0
        for index in range(len(lb)):
            value += np.exp2(index) * lb[index]
        lbp.append(value)
    lbp = np.array(lbp)
    lbp /= np.max(lbp)
    return lbp

def compute_laplace(vertices, faces, texels_gray, v_to_face):
    laplace = []

    for num in range(len(texels_gray)):
        one_ring = myutil.get_one_ring_vertex(v_to_face, num, faces)
        one_ring.remove(num)
        ini = np.zeros(3)
        for one in one_ring:
            ini += texels_gray[one] * vertices[one]
        curren_feature = myutil.norm(vertices[num] - ini)
        laplace.append(curren_feature)

    return laplace

def get_texels(data_name):
    # obj_path = "/home/simula/Dataset/textured_model/" + data_name + "/" + data_name + ".obj"
    obj_path = "E:\\mycodes\\Dataset\\textured_model\\" +data_name + "\\" + data_name + ".obj"
    obj_data = myutil.load_obj(obj_path)
    vertices, faces, uvs, uv_index = obj_data[0], obj_data[1], obj_data[2], obj_data[3]  # uv_index是和面相关的uv的index
    v_to_face = myutil.get_vertex_to_face(faces)

    # position move
    vertices2 = [vertex+[500, 0, 0] for vertex in vertices]
    vertices2 = np.array(vertices2)

    # mean_curvature
    mc_path = "E:\\mycodes\\Dataset\\textured_model\\" + data_name + "\\saliency.txt"
    file = open(mc_path)
    lines = file.readlines()
    mean_curvature = np.array([np.float(s) for s in lines])
    mean_curvature = exposure.equalize_hist(mean_curvature)

    # compute texel
    img_path = "E:\\mycodes\\Dataset\\textured_model\\bunny\\bunny-atlas.jpg"
    img = cv2.imread(img_path)
    img = rotation(img)

    v_to_uvs = myutil.get_v_to_uv(vertices, faces, uv_index)

    texel_uvs = [uvs[uv_set.pop()] for uv_set in v_to_uvs]

    texels = []
    texels_coord = []
    for uv in texel_uvs:
        x = round(np.float(uv[0])*1024)
        y = round(np.float(uv[1])*1024)
        texels_coord.append((x, y))
        pixel = img[y][x]
        item = np.array([pixel[0], pixel[1], pixel[2], 255], dtype=np.uint8)
        texels.append(item)

    texels_gray = [
        0.2989*texel[0]+0.5870*texel[1]+0.1140*texel[2] for texel in texels
    ]

    # lbp = compute_lbp(texels, faces, texels_gray, v_to_face)

    laplace = compute_laplace(vertices, faces, texels_gray, v_to_face)
    # mayavi_pcl(vertices, texels, mean_curvature)
    # mlab.show()

    return texels, vertices, faces, laplace


def get_local_spfh(vertices, faces, normals):
    spfh_list = []
    for num in range(len(vertices)):
        vertex = vertices[num]
        normal = normals[num]
        one_ring = myutil.get_one_ring_vertex(v_to_face, num, faces) # v_to_face, vertex_index, faces
        one_ring.remove(num)
        vertex_t_index_list = list(one_ring)
        for vertex_t_index in vertex_t_index_list:
            vertex_t = vertices[vertex_t_index]
            u = normal
            v = np.cross(vertex_t - vertex, u)
            w = np.cross(u, v)
            normal_t = normals[vertex_t_index]
            alpha = np.dot(v, normal_t)
            phi = np.dot(u, (vertex_t - vertex)/myutil.norm(vertex_t - vertex))
            theta = np.arctan([np.dot(w, normal_t), np.dot(u, normal_t)])
            features = [alpha, phi, theta[0], theta[1]]
            spfh_list.append(features)

    return spfh_list

def get_local_fpfh(vertices, faces, texels, spfh_list):
    fpfh_list = []
    texels_gray = [
        0.2989*texel[0]+0.5870*texel[1]+0.1140*texel[2] for texel in texels
    ]
    texels_gray = np.array(texels_gray)
    texels_gray /= np.max(texels_gray)
    # v_to_face = myutil.get_vertex_to_face(faces)
    for num in range(len(vertices)):
        one_ring = myutil.get_one_ring_vertex(v_to_face, num, faces)
        one_ring.remove(num)
        num_PV = np.array([spfh_list[num]])
        num_PM = num_PV.T * num_PV
        eigen_I = np.zeros(4*4).reshape([4, 4])
        evs = []
        for one_ring_i in one_ring:
            point_feature = spfh_list[one_ring_i]
            point_feature_vector = np.array([point_feature])

            point_feature_vector /= np.exp(texels_gray[num] - texels_gray[one_ring_i])
            point_feature_matrix = point_feature_vector.T * point_feature_vector
            # point_feature_matrix /= texels_gray[num] - texels_gray[one_ring_i]
            # point_feature_matrix /= np.exp(texels_gray[num] - texels_gray[one_ring_i])
            # eigen_I += point_feature_matrix
            eigen_value, eigen_vector = np.linalg.eig(point_feature_matrix)
            ev = np.mean(eigen_value)
            evs.append(np.real(ev))

        num_PM += eigen_I

        fpfh_list.append(np.mean(evs))

    return fpfh_list


def compute_extreme_point(vertices, faces, fpfh_list):
    # v_to_face = myutil.get_vertex_to_face(faces)
    extrem_point = []
    for num in range(len(vertices)):
        one_ring, one_segment = myutil.get_n_ring_vertex(v_to_face, num, faces, 5)
        one_ring.remove(num)
        extrem_yes = True
        for one_ring_i in one_ring:
            if fpfh_list[num] < fpfh_list[one_ring_i]:
                extrem_yes = False

        if extrem_yes is True:
            extrem_point.append(num)

    return extrem_point

def compute_gw(fpfh_list, num, ring, vertices, p_feature):

    p = vertices[num]
    dist_down = 0
    dist_up = 0
    for index in ring:
        pt = vertices[index]
        dist_down += np.exp(myutil.norm(p - pt))
        dist_up += np.exp(myutil.norm(p - pt))*p_feature[index]
    try:
        gw = dist_up / dist_down
        return gw
    except:
        return 1

def build_pyramid(fpfh_list, vertices, faces, p_feature):
    gw_one = []
    gw_two = []
    gw_thr = []
    gw_four = []
    gw_five = []
    gw_six = []

    v_to_face = myutil.get_vertex_to_face(faces)

    for num in range(len(vertices)):
        if num % 1000 == 0:
            print("current index = {}".format(num))

        one_ring, one_segment = myutil.get_n_ring_vertex(v_to_face, num, faces, 1)
        two_ring, two_segment = myutil.get_n_ring_vertex(v_to_face, num, faces, 2)
        three_ring, three_segment = myutil.get_n_ring_vertex(v_to_face, num, faces, 3)
        four_ring, four_segment = myutil.get_n_ring_vertex(v_to_face, num, faces, 4)
        five_ring, five_segment = myutil.get_n_ring_vertex(v_to_face, num, faces, 5)
        six_ring, six_segment = myutil.get_n_ring_vertex(v_to_face, num, faces, 6)

        for five in five_ring: six_ring.remove(five)
        for four in four_ring: five_ring.remove(four)
        for three in three_ring: four_ring.remove(three)
        for two in two_ring: three_ring.remove(two)
        one_ring.remove(num)

        pure_six = six_ring
        pure_five = five_ring
        pure_four = four_ring
        pure_three = three_ring
        pure_two = two_ring
        pure_one = one_ring

        gw_one.append(compute_gw(fpfh_list, num, pure_one, vertices, p_feature))
        # gw_two.append(compute_gw(fpfh_list, num, pure_two, vertices, p_feature))
        # gw_thr.append(compute_gw(fpfh_list, num, pure_three, vertices, p_feature))
        # gw_four.append(compute_gw(fpfh_list, num, pure_four, vertices, p_feature))
        # gw_five.append(compute_gw(fpfh_list, num, pure_five, vertices, p_feature))
        gw_six.append(compute_gw(fpfh_list, num, pure_six, vertices, p_feature))

    gw_one, gw_two, gw_thr, gw_four, gw_five, gw_six = \
        np.array(gw_one), \
        np.array(gw_two), \
        np.array(gw_thr), \
        np.array(gw_four), \
        np.array(gw_five), \
        np.array(gw_six)

    return gw

def main():

    data_name = "bunny"
    global v_to_face
    texels, vertices, faces, p_feature = get_texels(data_name)
    print("end the texels")

    # normals = myutil.compute_point_normal(data_name)
    fpfh_list = []
    # spfh_list = get_local_spfh(vertices, faces, normals)
    # fpfh_list = get_local_fpfh(vertices, faces, texels, spfh_list)
    # extrem_point = compute_extreme_point(vertices, faces, fpfh_list)
    print("end the local descriptor")

    myutil.mayavi_with_custom_point(vertices, faces, p_feature)
    mlab.show()
    exit(0)

    print("enter the pyramid")
    gw = build_pyramid(fpfh_list, vertices, faces, p_feature)

    write_vs(data_name+"_aug.txt", gw)

    myutil.mayavi_with_custom_point(vertices, faces, gw)
    mlab.show()


def fusion():

    data_name = "bunny"
    white = (1,1,1)
    texels, vertices, faces, lbp = get_texels(data_name)

    mv_vs = read_fs("bunny_vs.txt")
    aug_vs = read_fs("bunny_aug.txt")

    min_aug, max_aug = min(aug_vs), max(aug_vs)
    aug_vs = [aug_vs_item - min_aug for aug_vs_item in aug_vs]
    aug_vs = np.array(aug_vs)
    aug_vs /= max(aug_vs)

    mv_vs = np.array(mv_vs)
    aug_vs = np.array(aug_vs)
    print(len(mv_vs))
    print(aug_vs[:50])

    white = (1, 1, 1)
    black = (0, 0, 0)
    fig = mlab.figure(size=(512, 512 + 48), bgcolor=white)

    # fuse = mv_vs + aug_vs * 0.4
    fuse = aug_vs
    myutil.mayavi_with_custom_point(vertices, faces, aug_vs)

    write_vs('bunny_aug.txt', fuse)

    # vertices[:, 0] += 600
    # myutil.mayavi_with_custom_point(vertices, faces, fuse)
    mlab.show()


main()
# fusion()
# tip()