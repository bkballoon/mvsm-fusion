import numpy as np
from mayavi import mlab

def load_obj(filename):
    vertices_ = []
    faces_ = []
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
    vertices_ = np.array(vertices_)
    faces_ = np.array(faces_)
    return vertices_, faces_

def mayavi_model(vertices, faces, cell_data_custom):
    # cell_data_custom = [0.1 for i in range(99328)]
    mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                faces)
    cell_data = mesh.mlab_source.dataset.cell_data
    cell_data.scalars = cell_data_custom
    cell_data.scalars.name = "cell data"
    cell_data.update()
    mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='cell data')
    mlab.pipeline.surface(mesh2)
    mlab.show()
    return vertices, faces

def load_vertex_saliecy(path):
    vertex_saliecy_local = []
    file = open(path)
    for i in file.readlines():
        vertex_saliecy_local.append(float(i))
    return vertex_saliecy_local


def ground_truth_mesh_saliency():
    obj_path     = "/home/simula/Dataset/fixation_model/3DModels-Simplif/bunny.obj"
    obj_gt_path1 = "/home/simula/Dataset/fixation_model/FixationMaps/bunny_300norm.txt"
    obj_gt_path2 = "/home/simula/Dataset/fixation_model/FixationMaps/bunny_413norm.txt"
    obj_gt_path3 = "/home/simula/Dataset/fixation_model/FixationMaps/bunny_599norm.txt"

    vertices, faces = load_obj(obj_path)
    vertex_saliency1 = load_vertex_saliecy(obj_gt_path1)
    vertex_saliency2 = load_vertex_saliecy(obj_gt_path2)
    vertex_saliency3 = load_vertex_saliecy(obj_gt_path3)

    vertex_saliency = vertex_saliency3
    # for i in range(len(vertices)):
    #     vertex_saliency.append(np.mean([
    #         vertex_saliency1[i], vertex_saliency2[i], vertex_saliency3[i]
    #     ]))

    face_saliency = []
    for i in range(len(faces)):
        face_saliency.append([
            vertex_saliency[faces[i][0]],
            vertex_saliency[faces[i][1]],
            vertex_saliency[faces[i][2]],
        ])

    mayavi_model(vertices, faces, face_saliency)

def Lee_mesh_saliency():
    obj_path = "/home/simula/Dataset/fixation_model/3DModels-Simplif/bunny.obj"
    vertices, faces = load_obj(obj_path)

    Lee_obj_saliency = "/home/simula/Dataset/fixation_model/SaliencyAlgorithmMaps/bunny_Lee.CSV"
    lee_vertex_saliency = load_vertex_saliecy(Lee_obj_saliency)
    vertex_saliency = lee_vertex_saliency
    face_saliency = []
    for i in range(len(faces)):
        face_saliency.append([
            vertex_saliency[faces[i][0]],
            vertex_saliency[faces[i][1]],
            vertex_saliency[faces[i][2]],
        ])

    mayavi_model(vertices, faces, face_saliency)


Lee_mesh_saliency()
