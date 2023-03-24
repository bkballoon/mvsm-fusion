import numpy as np
from mayavi import mlab

def load_obj(filename):
    vertices_ = []
    faces_ = []
    img_textures_ = []
    face_textures_ = []

    for line in open(filename):
        if line.startswith('#'):
            continue
        values = line.split()
        if line == '':
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            v = [v[0], v[1], v[2]]
            vertices_.append(v)
        elif values[0] == 'vt':
            img_textures_.append([values[1], values[2]])
        elif values[0] == 'f':
            f = [int(values[1].split('/')[0]) - 1,
                 int(values[2].split('/')[0]) - 1,
                 int(values[3].split('/')[0]) - 1]
            faces_.append(f)
            try:
                face_texture = [int(values[1].split('/')[1]) - 1,
                                int(values[2].split('/')[1]) - 1,
                                int(values[3].split('/')[1]) - 1]
                face_textures_.append(face_texture)
            except:
                None

    vertices_ = np.array(vertices_)
    faces_ = np.array(faces_)
    img_textures_ = np.array(img_textures_)
    face_textures_ = np.array(face_textures_)
    print("this obj model has {} vertices, {} faces, {} uvs, {} uv_index".format(
        len(vertices_), len(faces_), len(img_textures_), len(face_textures_)))
    return [vertices_, faces_, img_textures_, face_textures_]


def mayavi_with_custom_face(vertices, faces, cell_data_custom, cm=None):
    # cell_data_custom = [0.1 for i in range(99328)]
    mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                faces)
    cell_data = mesh.mlab_source.dataset.cell_data
    cell_data.scalars = cell_data_custom
    cell_data.scalars.name = "cell data"
    cell_data.update()
    mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='cell data')
    if cm:
        mlab.pipeline.surface(mesh2, colormap=cm)
    else:
        mlab.pipeline.surface(mesh2)
    return mlab

def mayavi_with_custom_point(vertices, faces, cell_data_custom=None, cm=None):
    # mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2],
    #                             faces)
    # cell_data = mesh.mlab_source.dataset.point_data
    # cell_data.scalars = cell_data_custom
    # cell_data.scalars.name = "cell data"
    # cell_data.update()
    # mesh2 = mlab.pipeline.set_active_attribute(mesh, point_scalars='cell data')
    # mlab.pipeline.surface(mesh2)
    # mlab.show()
    if cell_data_custom is None:
        mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
        return mlab

    mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    point_data = mesh.mlab_source.dataset.point_data
    point_data.scalars = cell_data_custom
    point_data.scalars.name = "point data"
    point_data.update()
    mesh2 = mlab.pipeline.set_active_attribute(mesh, point_scalars="point data")
    if cm:
        mlab.pipeline.surface(mesh2, colormap=cm)
    else:
        mlab.pipeline.surface(mesh2)

    return mlab

# 每个点都有一个列表，该列表存储点相关的面的index从0开始
def get_vertex_to_face(faces):
    # load obj函数的face index都减了1
    v_to_face = dict()
    for index in range(len(faces)):
        face = faces[index]
        for vertex in face:
            if vertex not in v_to_face.keys():
                v_to_face[vertex] = [index]
            else:
                v_to_face[vertex].append(index)
    return v_to_face

# 获取点的一环内的邻域点
def get_one_ring_vertex(v_to_face, vertex_index, faces):
    face_list = v_to_face[vertex_index]
    # 一环邻域的点的容器
    point_set = set()
    for i in face_list:
        face_with_3_point = faces[i]
        for j in range(3):
            point_set.add(face_with_3_point[j])
    return point_set

# 构造线段集
def get_segment_set(origin_index, point_set, segment_set):
    for i in point_set:
        t = [origin_index, i]
        segment = '+'.join([str(i) for i in sorted(t)])
        segment_set.add(segment)

# 获取点的二环内的邻域点
def get_two_ring_vertex(v_to_face, vertex_index, faces):
    segment_set = set()
    point_set = get_one_ring_vertex(v_to_face, vertex_index, faces)
    get_segment_set(vertex_index, point_set, segment_set)

    two_ring_point_set = []
    for i in point_set:
        new_point_set = get_one_ring_vertex(v_to_face, i, faces)
        two_ring_point_set += list(new_point_set)
        get_segment_set(i, new_point_set, segment_set)

    two_ring_point_set = np.array(two_ring_point_set)
    two_ring_point_set = list(set(two_ring_point_set))
    return two_ring_point_set, segment_set

# 获取n环邻域内的点,迭代使用1环邻域内的点，这个函数大概率是没有问题的
def get_n_ring_vertex(v_to_face, vertex_index, faces, n):
    n_ring_point_index_set = set()
    n_ring_segment_set = set()

    new_ring = set()
    new_ring.add(vertex_index)
    for i in range(n):
        next_ring = set()
        for index in new_ring:
            segment_set = set()
            point_index_set = get_one_ring_vertex(v_to_face, index, faces)
            get_segment_set(index, point_index_set, segment_set)
            for k in point_index_set:
                n_ring_point_index_set.add(k)
                next_ring.add(k)
            next_ring.remove(index)
            for l in segment_set:
                n_ring_segment_set.add(l)
        new_ring.clear()
        new_ring = next_ring.copy()

    return n_ring_point_index_set, n_ring_segment_set


def norm(p):
    length = len(p)
    end = 0
    for i in range(length):
        end += p[i] * p[i]
    return np.sqrt(end)

def norm2(p):
    length = len(p)
    end = 0
    for i in range(length):
        end += p[i] * p[i]
    return end

# 两个点之间的距离
def distance_between_two_point(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    end = 0
    for i in range(len(p1)):
        end += np.power(p1[i] - p2[i], 2)
    return np.sqrt(end)

# 画球
def draw_sphere(ax, radius, origin):
    # 计算(0, 0, 0)圆心的球的表面的点，然后进行偏移
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # t, s = np.meshgrid(u, v)
    # x = np.cos(t) * np.sin(s) + origin[0]
    # y = np.sin(t) * np.sin(s) + origin[1]
    # z = np.cos(s) + origin[2]
    x = radius * np.outer(np.cos(u), np.sin(v)) + origin[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + origin[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + origin[2]

    ax.plot_surface(x, y, z, cmap='rainbow', alpha=0.1)
    return ax


# compute each point normal vector
def compute_point_normal(data_name):
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
                f = [int(values[1].split('/')[0]) - 1,
                     int(values[2].split('/')[0]) - 1,
                     int(values[3].split('/')[0]) - 1]
                faces_.append(f)
            elif values[0] == 'vn':
                n = [float(x) for x in values[1:4]]
                n = [n[0], n[1], n[2]]
                normals_.append(n)
        vertices_ = np.array(vertices_)
        faces_ = np.array(faces_)
        normals_ = np.array(normals_)
        return [vertices_, faces_, normals_]

    vertices, faces, nnnorm = load_obj("/home/simula/Dataset/textured_model/"+data_name+"/"+data_name+".obj")
    v_to_triangles = get_vertex_to_face(faces)
    normal_list = []
    # get each triangles normal and get the mean of all faces
    for i in range(len(vertices)):
        triangles_index = v_to_triangles[i]
        ns = np.array([0, 0, 0], dtype=np.float)
        for j in triangles_index:
            three_point = faces[j]
            # compute triangle normal based on the three point
            p1 = vertices[three_point[0]]
            p2 = vertices[three_point[1]]
            p3 = vertices[three_point[2]]
            p1p2 = np.array(p1) - np.array(p2)
            p1p3 = np.array(p1) - np.array(p3)
            n = np.cross(p1p2, p1p3)
            n = n / norm(n)
            ns += n
        ns = ns / len(triangles_index)
        normal_list.append(ns)

    # 然后将所有点的法向量写入到文件中，cpp在调用
    # file = open("/home/simula/Dataset/textured_model/tiger/normal.txt", 'w')
    # for i in normal_list:
    #     file.write(' '.join([str(j) for j in i]) + '\n')
    # file.close()

    return normal_list

#
def area_3p(p0, p1, p2):
    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    area = norm(np.cross(p0 - p1, p0 - p2))
    return area

# compute the area of the triangle
def area_of_triangle(vertices, three_p):
    p0 = vertices[three_p[0]]
    p1 = vertices[three_p[1]]
    p2 = vertices[three_p[2]]

    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    area = norm(np.cross(p0 - p1, p0 - p2))
    return area


# compute the barycentric coordinates
def barycenter(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    bary_center = (p1+p2+p3)/3
    area1 = area_3p(bary_center, p2, p3)
    area2 = area_3p(bary_center, p1, p3)
    area3 = area_3p(bary_center, p1, p2)
    area = (area1 + area2 + area3)
    return area1/area, area2/area, area3/area

# compute triangles related triangles
def compute_tri_with_tri(vertices, faces):
    v_to_tri = get_vertex_to_face(faces)
    tri_to_tris = []
    # 对任意一个三角面片，求任意两个顶点相关face的交集，然后得到面相关的3个面
    for face_i in range(len(faces)):
        triangle = faces[face_i]
        p1, p2, p3 = triangle[0], triangle[1], triangle[2]
        p1_to_tris = set(v_to_tri[p1])
        p2_to_tris = set(v_to_tri[p2])
        p3_to_tris = set(v_to_tri[p3])
        intersection1 = p1_to_tris.intersection(p2_to_tris)
        intersection2 = p1_to_tris.intersection(p3_to_tris)
        intersection3 = p2_to_tris.intersection(p3_to_tris)
        end_intersection = intersection1.union(intersection2).union(intersection3)
        end_intersection.remove(face_i)

        tri_to_tris.append(list(end_intersection))

    return tri_to_tris


# 计算顶点对应的uv值
def get_v_to_uv(vertices, faces, uv_index):
    v_to_uvs = []
    for i in range(len(vertices)):
        v_to_uvs.append(set())
    for i in range(len(faces)):
        face = faces[i]
        p1, p2, p3 = face[0], face[1], face[2]
        v_to_uvs[p1].add(uv_index[i][0])
        v_to_uvs[p2].add(uv_index[i][1])
        v_to_uvs[p3].add(uv_index[i][2])
    return v_to_uvs


# 计算点一环点的距离的平均
def get_v_normal_dist(v_to_face, vertices, faces):
    v_n_dist = []
    for i in range(len(vertices)):
        point_set = get_one_ring_vertex(v_to_face, i, faces)
        # 遍历point set，然后获取所有距离，归一化，然后取平均
        dist_list = []
        for point in point_set:
            vertex = vertices[point]
            dist = norm(vertex - vertices[i])
            dist_list.append(dist)
        # 归一化distlist
        dist_min, dist_max = min(dist_list), max(dist_list)
        dist_list /= (dist_max - dist_min)
        # 然后取平均即可
        dist_n_ave = np.mean(dist_list)
        v_n_dist.append(dist_n_ave)

    return v_n_dist

# 计算面片周围的平均归一化距离
def get_tri_normal_dist(tri_to_tris, vertices, faces):
    tri_to_n_dist = []
    for face_i in range(len(faces)):
        related_faces = tri_to_tris[face_i]
        center_list = []
        for r_face in related_faces:
            three_p = faces[r_face]
            p1, p2, p3 = vertices[three_p[0]], vertices[three_p[1]], vertices[three_p[2]]
            p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
            bary_center = (p1 + p2 + p3) / 3
            center_list.append(bary_center)
        dist_list = []
        current_face = faces[face_i]
        p1, p2, p3 = vertices[current_face[0]], vertices[current_face[1]], vertices[current_face[2]]
        current_center = (p1+p2+p3)/3
        for center in center_list:
            dist = distance_between_two_point(center, current_center)
            dist_list.append(dist)
        dist_list = np.array(dist_list)
        dist_aver = np.mean(dist_list/max(dist_list))
        tri_to_n_dist.append(dist_aver)

    return tri_to_n_dist

# 最简单的面积加权
def f2v(vertices, faces, face_saliency):
    # v_to_triangles
    v_to_triangles = get_vertex_to_face(faces)
    areas = []
    for face_i in range(len(faces)):
        area = area_of_triangle(vertices, faces[face_i])
        areas.append(area)

    # face saliency to vertex saliency based on the area of triangle
    vertex_saliency = np.zeros(len(vertices))
    for i in range(len(vertices)):
        tris = v_to_triangles[i]
        local_areas = []
        for tri_index in tris:
            local_areas.append(areas[tri_index])
            # vertex_saliency[i] += face_saliency[tri_index] * areas[tri_index]
            vertex_saliency[i] += face_saliency[tri_index]

        # vertex_saliency[i] = vertex_saliency[i] / (max(local_areas) - min(local_areas))
        vertex_saliency[i] = vertex_saliency[i] / len(tris)

    return vertex_saliency

def edge_to_pixel(faces, uv_index):
    edge_pixel = dict()
    # 对于每条边，确定其对应的顶点color，然后采样得到最后的结果
    for face_i in range(len(faces)):
        triangle = faces[face_i]
        triangle_uv_index = uv_index[face_i]

        for i in range(3):
            j = (i+1)%3
            if triangle[i] < triangle[j]:
                edge = str(triangle[i]) + "+" + str(triangle[j])
                pixel = str(triangle_uv_index[i]) + "+" + str(triangle_uv_index[j])
            else:
                edge = str(triangle[j]) + "+" + str(triangle[i])
                pixel = str(triangle_uv_index[j]) + "+" + str(triangle_uv_index[i])

            if edge not in edge_pixel.keys():
                edge_pixel[edge] = [pixel]
            else:
                edge_pixel[edge].append(pixel)

    return edge_pixel


# 色彩点云
def mayavi_point_cloud(vertices, colors):
    colors = np.array(colors)
    x = vertices[:, 0]/100
    y = vertices[:, 1]/100
    z = vertices[:, 2]/100
    n = len(vertices)  # number of points
    # x, y, z = np.random.random((3, n))
    rgba = np.random.randint(0, 256, size=(n, 4), dtype=np.uint8)
    rgba[:, -1] = 255  # no transparency
    rgba[:, 0] = colors[:, 0]
    rgba[:, 1] = colors[:, 1]
    rgba[:, 2] = colors[:, 2]

    pts = mlab.pipeline.scalar_scatter(x, y, z)  # plot the points
    pts.add_attribute(rgba, 'colors')  # assign the colors to each point
    pts.data.point_data.set_active_scalars('colors')
    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = 0.05  # set scaling for all the points
    g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size

    mlab.show()


# dataname = "tiger"
# # obj_path = "/home/simula/Dataset/textured_model/" + dataname +"/" + dataname + ".obj"
# obj_path = "D:\\project\\Dataset\\fixation_model\\3DModels-Simplif\\bunny.obj"
# obj_data = load_obj(obj_path)
# vertices, faces, uvs, uv_index = obj_data[0], obj_data[1], obj_data[2], obj_data[3]  # uv_index是和面相关的uv的index
# mayavi_with_custom_point(vertices, faces)
# mlab.show()

print("import the myutil")