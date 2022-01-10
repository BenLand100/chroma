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

    def build_part(self, save=True, rotation=None, displacement=None):
        seeker = Query()
        combined_mesh = None
        if displacement is None:
            displacement = np.zeros(3)
        if rotation is None:
            rotation = np.zeros(3)
        for key, entry in self.instructions.items():
            mesh_file = self.mesh_db.search(seeker.name == entry["mesh"])[0]["file"]
            mesh, pv_mesh = self.__assemble_mesh(mesh_file, entry["rotation"], entry["displacement"])
            if combined_mesh is None:
                combined_mesh = copy.deepcopy(pv_mesh)
            else:
                combined_mesh.merge(pv_mesh, inplace=True)
        global_center = combined_mesh.center
        for key, entry in self.instructions.items():
            mesh_file = self.mesh_db.search(seeker.name == entry["mesh"])[0]["file"]
            mesh, pv_mesh = self.__assemble_mesh(mesh_file, entry["rotation"], entry["displacement"], 
                                                 global_center, rotation, displacement)

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
                det_id = self.geometry.add_pmt(solid, rotation=np.identity(3), displacement=np.zeros(3))
                if save:
                    self.solid_dict[entry["mesh"]] = {
                        "solid": solid, 
                        "solid_id": det_id["solid_id"], 
                        "channel": det_id["channel_index"]
                    }
            else:
                solid_id = self.geometry.add_solid(solid, rotation=np.identity(3), displacement=np.zeros(3))
                if save:
                    self.solid_dict[entry["mesh"]] = {"solid": solid, "solid_id": solid_id, "channel": None}

            if save:
                self.mesh_dict[entry["mesh"]] = pv_mesh
                self.inner_mat_dict[entry["mesh"]] = inner_material
                self.outer_mat_dict[entry["mesh"]] = outer_material
                self.surface_dict[entry["mesh"]] = surface
                self.detectors[entry["mesh"]] = entry["detector"]
                self.rotations[entry["mesh"]] = entry["rotation"]
                self.displacements[entry["mesh"]] = entry["displacement"]

        if rotation is not None:
            combined_mesh.rotate_x(rotation[0], point=global_center)
            combined_mesh.rotate_y(rotation[1], point=global_center)
            combined_mesh.rotate_z(rotation[2], point=global_center)
        if displacement is not None:
            combined_mesh.translate(displacement)

        return combined_mesh, self.solid_dict

    def return_mesh_files(self):
        file_list = []
        seeker = Query()
        for key, entry in self.instructions.items():
            mesh_entry = self.mesh_db.search(seeker.name == entry["mesh"])[0]
            mesh_name = mesh_entry["name"]
            mesh_file = mesh_entry["file"]
            mesh_color = mesh_entry["color"]
            displacement = None
            rotation = None
            if entry["displacement"] is None:
                displacement = np.zeros(3)
            else:
                displacement = np.array(entry["displacement"])
            if entry["rotation"] is None:
                rotation = np.zeros(3)
            else:
                rotation = np.array(entry["rotation"])
            file_list.append((mesh_name, mesh_file, mesh_color,
                              displacement, rotation))
        return file_list

    @staticmethod
    def __check_db_search(results, search_value, db_name):
        if len(results) > 1:
            raise ValueError(f"{len(results)} found instead of 1 for search value {search_value} in {db_name}")
        elif len(results) == 0:
            raise ValueError(f"Search value {search_value} has no occurences in {db_name}")
        else:
            return results[0]
            
    @staticmethod
    def __assemble_mesh(file_name, rotation, displacement, global_center=None, global_rot=None, global_dist=None):
        mesh_data = pv.read(file_name)
        center = mesh_data.center_of_mass
        if rotation is not None:
            mesh_data.rotate_x(rotation[0], point=center)
            mesh_data.rotate_y(rotation[1], point=center)
            mesh_data.rotate_z(rotation[2], point=center)
        if displacement is not None:
            mesh_data.translate(displacement)
        if global_rot is not None:
            if global_center is None:
                global_center = np.zeros(3)
            mesh_data.rotate_x(global_rot[0], point=global_center)
            mesh_data.rotate_y(global_rot[1], point=global_center)
            mesh_data.rotate_z(global_rot[2], point=global_center)
        if global_dist is not None:
            mesh_data.translate(global_dist)
        vertices = mesh_data.points
        triangles = mesh_data.faces.reshape(-1, 4)[:, 1:]
        return Mesh(vertices, triangles), mesh_data

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
        self.all_parts_dict = {}
        self.all_solids_dict = {}

    def build_components(self, save=True, save_all_parts=False):
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
            part_mesh, solid_dict = part_builder.build_part(save, comp["displacement"], comp["rotation"])
            if save:
                self.comp_dict[comp["name"]] = part_mesh
            if save_all_parts:
                self.all_parts_dict[comp["name"]] = part_builder.mesh_dict
                self.all_solids_dict.update(solid_dict)

    def output_mesh_files(self, path, all_mesh=False):
        if all_mesh:
            for comp_name, part_dict in self.all_parts_dict.items():
                for part_name, part_mesh in part_dict.items():
                    destination = os.path.join(path, comp_name, f"{part_name}.stl")
                    if not os.path.exists(os.path.join(path, comp_name)):
                        os.makedirs(os.path.join(path, comp_name))
                    part_mesh.save(destination)
        else:
            for comp_name, comp_mesh in self.comp_dict():
                destination = os.path.join(path, f"{comp_name}.stl")
                if not os.path.exists(path):
                    os.makedirs(path)
                comp_mesh.save(destination)


    def plot_component_mesh(self, plotter, global_args=None, dict_args=None):
        if global_args is None:
            global_args = {}
        if dict_args is None:
            dict_args = {}
        for name, mesh in self.comp_dict.items():
            if name in dict_args.keys():
                settings = global_args
                settings.update(dict_args[name])
                plotter.add_mesh(mesh, **settings)
            else:
                plotter.add_mesh(mesh, **global_args)
