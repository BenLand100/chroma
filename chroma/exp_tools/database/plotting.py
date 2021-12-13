import os
import matplotlib.pyplot as plt
import pyvista as pv

def plot_1D(ax, value, label):
    ax.bar(value, height=1, width=0.01*value)
    ax.set_xlabel(label)


def plot_2D(ax, x, y, xlabel, ylabel):
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def db_visualizer(database, save_name=None, save_path=None, figsize=None):
    all_entries = database.all()
    for entry in all_entries:
        keys = list(entry.keys())
        fig, ax = plt.subplots(len(keys)-1, 1, figsize=figsize)
        title = entry["name"]
        fig.suptitle(entry["name"])
        for i, key in enumerate(keys):
            if key == "name":
                continue
            else:
                key_data = entry[key]
                if isinstance(key_data, float) | isinstance(key_data, int):
                    plot_1D(ax[i-1], key_data, key)
                elif isinstance(key_data, list):
                    if isinstance(key_data[0][0], float):
                        plot_2D(ax[i-1], key_data[1], key_data[0], "Wavelength (nm)", key)
                    else:
                        for data in key_data:
                            plot_2D(ax[i-1], data[1], data[0], "Wavelength (nm)", key)
        plt.tight_layout()
        if save_path:
            destination = os.path.join(save_path, f"{title}_{save_name}.png")
            plt.savefig(destination)


def mesh_visualizer(database, save_name=None, save_path=None):
    all_entries = database.all()
    for entry in all_entries:
        mesh = pv.read(entry["file"])
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        if save_name:
            if save_path:
                plotter.show(screenshot=os.path.join(save_path, save_name))
            else:
                plotter.show(screenshot=os.path.join(os.getcwd(), save_name))