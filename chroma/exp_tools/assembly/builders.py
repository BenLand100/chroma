import json
import numpy as np
import pyvista as pv

from tinydb import TinyDB, Query
from chroma.geometry import Material, Surface, Mesh, Solid
from chroma.exp_tools.physics.colors import rgb_surface


class PartBuilder:

    def __init__(self, geometry, instructions, mesh_db, material_db, surface_db):
        self.geometry = geometry
        if isinstance(instructions, TinyDB):
            self.instructions = instructions
        else:
            with open(instructions, "r") as json_file:
                self.instructions = json.load(json_file)
        self.mesh_db = mesh_db
        self.material_db = material_db
        self.surface_db = surface_db

        self.material_dict = {}
        self.surface_dict = {}
        self.solid_dict = {}

    def build_part(self):
        seeker = Query()
        for key, entry in self.instructions.items():
            mesh_file = self.mesh_db.search(seeker.name == entry["mesh"])[0]["file"]
            mesh = self.__assemble_mesh(mesh_file)

            inside_mat_dict = self.material_db.search(seeker.name == entry["material_in"]["name"])[0]
            outside_mat_dict = self.material_db.search(seeker.name == entry["material_out"]["name"])[0]
            surface_dict = self.surface_db.search(seeker.name == entry["surface"]["name"])[0]
            inner_material = self.__assemble_material(inside_mat_dict, entry["material_in"])
            outer_material = self.__assemble_material(outside_mat_dict, entry["material_out"])

            surface = self.__assemble_surface(surface_dict, entry["surface"])

            solid = Solid(mesh, material1=inner_material, material2=outer_material, surface=surface)
            if entry["detector"] is True:
                self.geometry.add_pmt(solid, rotation=entry["rotation"], displacement=entry["displacement"])
            else:
                self.geometry.add_solid(solid, rotation=entry["rotation"], displacement=entry["displacement"])

            self.material_dict[entry["material_in"]["name"]] = inner_material
            self.surface_dict[entry["surface"]["name"]] = surface
            self.solid_dict[self.instructions[key]["mesh"]] = {solid}

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
            if key == "name": continue
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
