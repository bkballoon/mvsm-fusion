# 画统一尺度算法的流程图
import numpy as np
from mayavi import mlab
import cv2
from matplotlib import pyplot as plt
import yang.myutil as myutil

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


def mayavi_with_custom_face(vertices, faces, cell_data_custom):
    # cell_data_custom = [0.1 for i in range(99328)]
    mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                faces)
    cell_data = mesh.mlab_source.dataset.cell_data
    cell_data.scalars = cell_data_custom
    cell_data.scalars.name = "cell data"
    cell_data.update()
    mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='cell data')
    mlab.pipeline.surface(mesh2)
    # mlab.show()
    return mlab

def rgb_to_face(rgb):
    k = 255 * 255 * rgb[0] + 255 * rgb[1] + 1 * rgb[2]
    return k

# 画Sortting部分的流程示意图
def draw_figure_one():
    rgb_pic_name = []
    number = 2
    dis = 90
    for i in range(number):
        path = "/home/simula/Pic/paper/video0/" + str(dis * i + 15) + ".png"
        rgb_pic_name.append(path)


    index_color_pic_name = []
    for i in range(number):
        path = "/home/simula/Pic/paper/video1/" + str(dis*i + 15) + ".png"
        index_color_pic_name.append(path)

    vertices, faces, nnorms = load_obj("/home/simula/Dataset/textured_model/bunny/bunny.obj")

    face_color = np.zeros(len(faces))

    for i in index_color_pic_name:
        face_list = []
        index_img = cv2.imread(i)
        h, w, _ = index_img.shape
        for x in range(h):
            for y in range(w):
                pixel = index_img[x][y]
                face_index = rgb_to_face(pixel) - 1
                if 0 <= face_index < len(faces):
                    face_list.append(face_index)

        face_list = list(set(face_list))

        for face_i in face_list:
            face_color[face_i] += 0.1

    mlab.figure(figure=(400, 400))
    myutil.mayavi_with_custom_face(vertices, faces, face_color)
    mlab.show()


def show_observation(number):
    path = str(number) + "_vs_salicon.txt"
    all_lines = open(path).readlines()
    r_list = []
    r = []
    all_lines = [float(i) for i in all_lines]
    for i in all_lines:
        if i != 99999:
            r.append(i)
        else:
            r_list.append(r)
            r = []
    print(len(r_list))

    case = 1
    max_taken = []
    first_three = []
    patch_list = []
    for r in r_list:
        print("---> case is ", case)
        case += 1
        r_min = min(r)
        r_max = max(r)

        dis = 200
        patch = (r_max - r_min)/dis
        r_index = [int((r_i - r_min)/patch) for r_i in r]
        s = np.zeros(dis+1)
        for r_i in r_index:
            s[r_i] += 1

        for s_i in range(len(s)):
            s[s_i] /= len(r)

        s_sort = sorted(s)

        index = np.where(s == s_sort[-1])[0]
        print("max taken is ", s[index])
        print("first three is ", s[index] + s[index+1] + s[index-1])
        print("patch is ", patch)

        max_taken.append(s[index][0])
        first_three.append(s[index][0] + s[index+1][0] + s[index-1][0])
        patch_list.append(patch)

        n, bins, patches = plt.hist(r, bins=100)
        plt.xlabel("Rate")
        plt.ylabel("Frequency")
        plt.title("Rate histogram")

        print(bins)
        print(n)
        plt.show()

    print()

    print(max_taken)
    print(first_three)
    print(patch_list)

    print(np.mean(first_three))
    print(np.mean(patch_list))

    # x = [str(i) for i in range(number+1)]
    # plt.plot(x, max_taken)
    # plt.plot(x, first_three)
    # plt.plot(x, patch_list)
    # plt.show()

show_observation(20)
