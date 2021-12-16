import os, glob
import numpy as np
from tinydb import TinyDB, Query


def mesh_db_template(file_path, db_name, db_path=None):
    if db_path:
        new_db = TinyDB(os.path.join(db_path, db_name), indent=4)
    else:
        new_db = TinyDB(db_name, indent=4)
    file_list = glob.glob(os.path.join(file_path, "*"))
    file_list.sort()
    for file in file_list:
        new_db.insert({"name": "name", "file": file, "color": None})


def update_mesh_db(file_path, db_name, db_path=None):
    if db_path:
        db = TinyDB(os.path.join(db_path, db_name), indent=4)
    else:
        db = TinyDB(db_name, indent=4)
    file_list = glob.glob(os.path.join(file_path, "*"))
    file_list.sort()
    db_search = Query()
    full_db = db.all()
    for i, file in enumerate(file_list):
        db.update({"file": file}, db_search.name == full_db[i]["name"])


def read_db(database, name, attribute):
    searcher = Query()
    entry = database.search(searcher.name == name)
    return entry[0][attribute]


def update_db(database, name, new_entry):
    searcher = Query()
    database.update(new_entry, searcher.name == name)


def rgb_surface(rgb_array):
    inverse_array = np.array([255, 255, 255]) - np.array(rgb_array)
    norm_rgb = np.array(rgb_array) / 255
    norm_rgb_inverse = inverse_array / 255
    wavelengths = np.array([60, 420, 480, 550, 580, 1000])
    reflectance = np.concatenate(([0], norm_rgb, [0]))
    absorb = np.concatenate(([1], norm_rgb_inverse, [1]))
    return reflectance, absorb, wavelengths