import json
import copy
import pyvista as pv

from chroma.geometry import Material, Surface, Solid, Mesh
from chroma.loader import create_geometry_from_obj


class GeometryLoader(object):
    
    def __init__(self, geo_dict=None, material_loader=None, surface_loader=None):
        self.geo_dict = geo_dict
        self.material_loader = material_loader
        self.surface_loader = surface_loader
        
    def create_geo_dict(self, file_list):
        output_dict = {}
        entry = {
            "file": None, 
            "inner_material": None, 
            "outer_material": None, 
            "surface": None, 
            "rotation": None, 
            "displacement": None, 
            "is_detector": False
        }

        for i, file_name in enumerate(file_list):
            output_dict[i] = copy.deepcopy(entry)
            output_dict[i]["file"] = file_name
        self.geo_dict = output_dict
        
    def create_geometry(self, output_geo):
        det_info = []
            
        for idx in self.geo_dict.keys():
            inner_material = self.assemble_material(idx, "material", "inner_material")
            outer_material = self.assemble_material(idx, "material", "outer_material")
            surface = self.assemble_material(idx, "surface", "surface")
            mesh = self._assemble_mesh(self.geo_dict[idx]["file"])
            solid = Solid(mesh, inner_material, outer_material, surface)
            
            if self.geo_dict[idx]["is_detector"]:
                if isinstance(self.geo_dict[idx]["displacement"], list):
                    rotation_list = self.geo_dict[idx]["rotation"]
                    for i, displace in enumerate(self.geo_dict[idx]["displacement"]):
                        rotation = None
                        if rotation_list is None:
                            rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                        else:
                            rotation = self.geo_dict[idx]["rotation"][i]
                        info = output_geo.add_pmt(solid, rotation=rotation, displacement=displace)
                        det_info.append(info)
                else:
                    info = output_geo.add_pmt(
                        solid, 
                        rotation=self.geo_dict[idx]["rotation"],
                        displacement=self.geo_dict[idx]["displacement"]
                    )
                    det_info.append(info)
            else:
                if isinstance(self.geo_dict[idx]["displacement"], list):
                    rotation_list = self.geo_dict[idx]["rotation"]
                    for i, displace in enumerate(self.geo_dict[idx]["displacement"]):
                        rotation = None
                        if rotation_list is None:
                            rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                        else:
                            rotation = self.geo_dict[idx]["rotation"][i]
                        output_geo.add_solid(solid, rotation=rotation, displacement=displace)
                else:
                    output_geo.add_solid(solid, rotation=self.geo_dict[idx]["rotation"], displacement=self.geo_dict[idx]["displacement"])
        
        return create_geometry_from_obj(output_geo), det_info
            
    def save_geo_dict(self, save_file):
        with open(save_file, "w+") as json_file:
            json.dump(self.geo_dict, json_file, indent=4)
        
    def load_geo_dict(self, input_file):
        with open(input_file, "r") as json_file:
            self.geo_dict = json.load(json_file)
        
    @staticmethod
    def _assemble_mesh(file_name):
        mesh_file = pv.read(file_name)
        vertices = mesh_file.points
        triangles = mesh_file.faces.reshape(-1, 4)[:, 1:]
        return Mesh(vertices, triangles)

    def assemble_material(self, idx, loader, material_name):
    	geo_dict_element = self.geo_dict[idx][material_name]
    	output_material = None
    	if isinstance(geo_dict_element, list):
    		output_material = []
    		for name in geo_dict_element:
    			if loader == "material":
    				output_material.append(self.material_loader.object_dict[name])
    			elif loader == "surface":
    				output_material.append(self.surface_loader.object_dict[name])
    			else:
    				print("Unknown loader: "+str(loader))
    	else:
    		if loader == "material":
    			output_material = self.material_loader.object_dict[self.geo_dict[idx][material_name]]
    		elif loader == "surface":
    			output_material = self.surface_loader.object_dict[self.geo_dict[idx][material_name]]
    	return output_material

class MaterialLoader(object):
    
    def __init__(self, material_dict=None):
        self.material_dict = material_dict
        self.object_dict = {}
        
    def create_dict(self, material_list):
        output_dict = {}
        entry = {"refractive_index": None, "absorption_length": None, "scattering_length": None, "density": None}
        for i, material in enumerate(material_list):
            output_dict[material] = copy.deepcopy(entry)
        self.material_dict = output_dict
        
    def create_objects(self):
        for material in self.material_dict.keys():
            self.object_dict[material] = Material(material)
            for prop in self.material_dict[material].keys():
                value = self.material_dict[material][prop]
                if isinstance(value, list):
                    self.object_dict[material].set(prop, value[0], value[1])
                else:
                    self.object_dict[material].set(prop, value)
        
    def __getitem__(self, material):
        return self.object_dict[material]
    
    def __setitem__(self, idx, value):
        material, prop = idx
        self.material_dict[material][prop] = value
        
    def save_mat_dict(self, save_file):
        with open(save_file, "w+") as json_file:
            json.dump(self.material_dict, json_file, indent=4)
        
    def load_mat_dict(self, input_file):
        with open(input_file, "r") as json_file:
            self.material_dict = json.load(json_file)
        
class SurfaceLoader(MaterialLoader):
    
    def __init__(self, material_dict=None):
        super().__init__()
        
    def create_dict(self, surface_list):
        output_dict = {}
        entry = {"detect": 0, "absorb": 0, "reemit": 0, "reflect_diffuse": 0, "reflect_specular": 0}
        for i, surface in enumerate(surface_list):
            output_dict[surface] = copy.deepcopy(entry)
        self.material_dict = output_dict
        
    def create_objects(self):
        for material in self.material_dict.keys():
            self.object_dict[material] = Surface(material)
            for prop in self.material_dict[material].keys():
                value = self.material_dict[material][prop]
                if isinstance(value, list):
                    self.object_dict[material].set(prop, value[0], value[1])
                else:
                    self.object_dict[material].set(prop, value)
