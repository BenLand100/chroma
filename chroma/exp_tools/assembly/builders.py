import os
import copy
import json
import numpy as np
import pyvista as pv

from tinydb import TinyDB, Query
from chroma.geometry import Material, Surface, Mesh, Solid
from chroma.exp_tools.physics.colors import rgb_surface


class PartBuilder:

    def __init__(self, geometry, instructions, mesh_db, material_db, surface_db):
        self.geometry = geometry
        if isinstance(instructions, dict):
            self.instructions = instructions
        else:
            with open(instructions, "r") as json_file:
                self.instructions = json.load(json_file)
        self.mesh_db = mesh_db
        self.material_db = material_db
        self.surface_db = surface_db

        self.mesh_dict = {}
        self.inner_mat_dict = {}
        self.outer_mat_dict = {}
        self.surface_dict = {}
        self.solid_dict = {}
        self.detectors = {}
        self.rotations = {}
        self.displacements = {}

    def build_part(self, save=True, displacement=None, rotation=None):
        seeker = Query()
        if displacement is None:
            displacement = np.array([0, 0, 0])
        if rotation is None:
            rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for key, entry in self.instructions.items():
            mesh_file = self.mesh_db.search(seeker.name == entry["mesh"])[0]["file"]
            mesh = self.__assemble_mesh(mesh_file)

            mat_search_in = self.material_db.search(seeker.name == entry["material_in"]["name"])
            mat_search_out = self.material_db.search(seeker.name == entry["material_out"]["name"])
            surf_search = self.surface_db.search(seeker.name == entry["surface"]["name"])

            inside_mat_dict = self.__check_db_search(mat_search_in, entry["material_in"]["name"], "materials")
            outside_mat_dict = self.__check_db_search(mat_search_out, entry["material_out"]["name"], "materials")
            surface_dict = self.__check_db_search(surf_search, entry["surface"]["name"], "surfaces")

            inner_material = self.__assemble_material(inside_mat_dict, entry["material_in"])
            outer_material = self.__assemble_material(outside_mat_dict, entry["material_out"])

            surface = self.__assemble_surface(surface_dict, entry["surface"])

            solid = Solid(mesh, material1=inner_material, material2=outer_material, surface=surface)
            if entry["detector"] is True:
                det_rotation = entry["rotation"]
                det_displace = entry["displacement"]
                if det_rotation is None:
                    det_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                if det_displace is None:
                    det_displace = [0, 0, 0]
                det_rotation = rotation * np.array(det_rotation)
                det_displace = displacement + det_displace
                det_id = self.geometry.add_pmt(solid, rotation=det_rotation, displacement=det_displace)
                if save:
                    self.solid_dict[det_id["solid_id"]] = {
                        "solid": solid, 
                        "name": entry["mesh"], 
                        "channel": det_id["channel_index"]
                    }
            else:
                solid_rotate = entry["rotation"]
                solid_displace = entry["displacement"]
                if entry["rotation"] is None:
                    solid_rotate = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                if entry["displacement"] is None:
                    solid_displace = np.array([0, 0, 0])
                
                solid_rotate = rotation * solid_rotate
                solid_displace = displacement + solid_displace

                solid_id = self.geometry.add_solid(solid, rotation=solid_rotate, displacement=solid_displace)
                if save:
                    self.solid_dict[solid_id] = {"solid": solid, "name": entry["mesh"], "channel": None}
            
            if save:
                self.mesh_dict[entry["mesh"]] = mesh
                self.inner_mat_dict[entry["mesh"]] = inner_material
                self.outer_mat_dict[entry["mesh"]] = outer_material
                self.surface_dict[entry["mesh"]] = surface
                self.detectors[entry["mesh"]] = entry["detector"]
                self.rotations[entry["mesh"]] = entry["rotation"]
                self.displacements[entry["mesh"]] = entry["displacement"]

    def return_mesh_files(self):
        file_list = []
        seeker = Query()
        for key, entry in self.instructions.items():
            mesh_entry = self.mesh_db.search(seeker.name == entry["mesh"])[0]
            mesh_name = mesh_entry["name"]
            mesh_file = mesh_entry["file"]
            mesh_color = mesh_entry["color"]
            file_list.append((mesh_name, mesh_file, mesh_color))
        return file_list

    # TODO: test is function is needed
    # def rebuild_part(self):
    #     if isinstance(self.geometry, Detector):
    #         self.geometry = Detector()
    #     else:
    #         self.geometry = Geometry()
    #     for key, solid in self.solid_dict.items():
    #         mesh = self.mesh_dict[solid["name"]]
    #         inner_mat = copy.deepcopy(self.inner_mat_dict[solid["name"]])
    #         outer_mat = copy.deepcopy(self.outer_mat_dict[solid["name"]])
    #         surface = copy.deepcopy(self.surface_dict[solid["name"]])
    #         new_solid = Solid(mesh, material1=inner_mat, material2=outer_mat, surface=surface)
    #         if self.detectors[solid["name"]]:
    #             rotation = self.rotations[solid["name"]]
    #             displacement = self.displacements[solid["name"]]
    #             if rotation is None:
    #                 rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #             if displacement is None:
    #                 displacement = [0, 0, 0]
    #             det_id = self.geometry.add_pmt(new_solid, 
    #                                            rotation=rotation, 
    #                                            displacement=displacement)
    #             self.solid_dict[det_id["solid_id"]] = {
    #                 "solid": new_solid, 
    #                 "name": solid["name"], 
    #                 "channel": det_id["channel_index"]
    #             }
    #         else:
    #             solid_id = self.geometry.add_solid(new_solid, rotation=self.rotations[solid["name"]], 
    #                                                displacement=self.displacements[solid["name"]])
    #             self.solid_dict[solid_id] = {"solid": new_solid, "name": solid["name"], "channel": None}

    @staticmethod
    def __check_db_search(results, search_value, db_name):
        if len(results) > 1:
            raise ValueError(f"{len(results)} found instead of 1 for search value {search_value} in {db_name}")
        elif len(results) == 0:
            raise ValueError(f"Search value {search_value} has no occurences in {db_name}")
        else:
            return results[0]
            
    @staticmethod
    def __assemble_mesh(file_name):
        mesh_data = pv.read(file_name)
        vertices = mesh_data.points
        triangles = mesh_data.faces.reshape(-1, 4)[:, 1:]
        return Mesh(vertices, triangles)

    @staticmethod
    def __assemble_material(mat_dict, instruct_entry):
        material = Material()
        for key, value in mat_dict.items():
            if key == "name":
                material.name = key
                continue
            if isinstance(value, list):
                material.set(key, value[0], wavelengths=value[1])
            else:
                material.set(key, value)

        for key, value in instruct_entry.items():
            if (key in dir(material)) & (key != "name"):
                if isinstance(value, list):
                    material.set(key, value[0], wavelengths=value[1])
                else:
                    material.set(key, value)

        return material

    @staticmethod
    def __assemble_surface(surf_dict, instruct_entry):
        surface = Surface()
        for key, value in surf_dict.items():
            if key == "name":
                surface.name = value
                continue
            if key == "reflectance":
                if isinstance(value, list):
                    surface.set("reflect_diffuse", 
                                np.array(value[0])*instruct_entry["diffuse"], 
                                wavelengths=value[1])
                    surface.set("reflect_specular", 
                                np.array(value[0])*instruct_entry["specular"], 
                                wavelengths=value[1])
                else:
                    surface.set("reflect_diffuse", value*instruct_entry["diffuse"])
                    surface.set("reflect_specular", value*instruct_entry["specular"])
            else:
                if isinstance(value, list):
                    surface.set(key, value[0], wavelengths=value[1])
                else:
                    surface.set(key, value)

        for key, value in instruct_entry.items():
            if (key in dir(surface)) & (key != "name"):
                if isinstance(value, list):
                    surface.set(key, value[0], wavelengths=value[1])
                else:
                    surface.set(key, value)
        
        if "rgb" in instruct_entry.keys():
            reflect, absorb, wavelengths = rgb_surface(instruct_entry["rgb"])
            surface.set("reflect_diffuse", reflect*instruct_entry["diffuse"], wavelengths=wavelengths)
            surface.set("reflect_specular", reflect*instruct_entry["specular"], wavelengths=wavelengths)
            surface.set("absorb", absorb, wavelengths=wavelengths)

        return surface


class ComponentBuilder:

    def __init__(self, geometry, instructions):
        self.geometry = geometry
        if isinstance(instructions, dict):
            self.instructions = instructions
        else:
            with open(instructions, "r") as json_file:
                self.instructions = json.load(json_file)

        self.mesh_db = TinyDB(self.instructions["mesh_db"])
        self.material_db = TinyDB(self.instructions["material_db"])
        self.surface_db = TinyDB(self.instructions["surface_db"])
        
        self.comp_dict = {}

    def build_components(self, save=True):
        for key, comp in self.instructions["components"].items():
            mesh_db_override = self.mesh_db
            instructions = None
            if "mesh_db" in comp.keys():
                mesh_db_override = comp["mesh_db"]
            part_instructs = os.path.join(self.instructions["parts_path"], comp["file"])
            with open(part_instructs, "r") as json_file:
                instructions = json.load(json_file)
            if "overrides" in comp.keys():
                for key, value in comp["overrides"].items():
                    for prop, attribute in comp["overrides"][key].items():
                        instructions[key][prop] = attribute
            part_builder = PartBuilder(self.geometry, instructions, 
                                       mesh_db_override, self.material_db, self.surface_db)
            part_builder.build_part(save, comp["displacement"], comp["rotation"])
            if save:
                self.comp_dict[comp["name"]] = part_builder

    def return_mesh_files(self):
        file_list = []
        for key, comp in self.instructions["components"].items():
            mesh_db_override = self.mesh_db
            instructions = None
            if "mesh_db" in comp.keys():
                mesh_db_override = comp["mesh_db"]
            part_instructs = os.path.join(self.instructions["parts_path"], comp["file"])
            with open(part_instructs, "r") as json_file:
                instructions = json.load(json_file)
            part_builder = PartBuilder(self.geometry, instructions, 
                                       mesh_db_override, self.material_db, self.surface_db)
            mesh_files = part_builder.return_mesh_files()
            file_list.extend(mesh_files)
        return file_list


def plot_mesh_files(mesh_file_list, plotter, global_args=None, dict_args=None):
    if global_args is None:
        global_args = []
    if dict_args is None:
        dict_args = []
    for name, file, color in mesh_file_list:
        mesh = pv.read(file)
        if name in dict_args.keys():
            settings = copy.deepcopy(dict_args[name])
            settings.update(global_args)
            plotter.add_mesh(mesh, color=color, **settings)
        else:
            plotter.add_mesh(mesh, color=color, **global_args)
