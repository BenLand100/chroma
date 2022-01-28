import numpy as np
import pyvista as pv


class PhotonPathTracer:

    def __init__(self, step_photons, step_photon_ids, photon_ids=None):
        self.step_photons = step_photons
        self.step_photon_ids = step_photon_ids
        self.photon_dict = {}
        if photon_ids is not None:
            self.get_hit_points(photon_ids)

    def get_hit_points(self, photon_ids):
        for i, step_photon in enumerate(self.step_photons):
            all_photon_ids = self.step_photon_ids[i]
            for photon_id in photon_ids:
                indices = np.where(all_photon_ids == photon_id)
                if len(indices[0]) == 0:
                    continue
                hit_position = step_photon.pos[indices[0][0]]
                if photon_id not in self.photon_dict.keys():
                    self.photon_dict[photon_id] = []
                self.photon_dict[photon_id].append(hit_position)

    def add_to_plotter(self, plotter, color=None, opacity=0.25):
        for photon_id, positions in self.photon_dict.items():
            line_poly_data = pv.lines_from_points(np.array(positions))
            plotter.add_mesh(line_poly_data, color=color, opacity=opacity)


def photon_hitmap(mesh, step_photons):
    triangle_values = np.zeros(len(mesh.triangles))
    for i, step in enumerate(step_photons):
        if i == 0: continue
        hit_triangles = step.last_hit_triangles
        for idx in hit_triangles:
            tri_vertices = mesh.triangles[idx]
            vec_a = mesh.vertices[tri_vertices[2]] - mesh.vertices[tri_vertices[0]]
            vec_b = mesh.vertices[tri_vertices[1]] - mesh.vertices[tri_vertices[0]]
            area_vec = np.cross(vec_a, vec_b)
            triangle_values[idx] += 1 / np.linalg.norm(area_vec)
    return triangle_values