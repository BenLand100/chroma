from chroma.gpu.detector import GPUDetector
from chroma.gpu.photon import GPUPhotons
from chroma.gpu.tools import get_rng_states

import chroma.exp_tools.physics.functions as func_dir

import os, time, glob
import h5py
import tqdm
import numpy as np


def tqdm_range(start, stop, step=1, verbose=True, text=None, bar_length=40, position=0):
    hide_bar = True
    if verbose:
        hide_bar = False
    bar_format = f"{{l_bar}}{{bar:{bar_length}}}{{r_bar}}{{bar:{-bar_length}b}}"

    return tqdm.trange(start, stop, step, position=position, disable=hide_bar, desc=text, bar_format=bar_format, leave=True)


def tqdm_it(iterable, verbose=True, text=None, bar_length=40, position=0, enum=False):
    hide_bar = True
    if verbose:
        hide_bar = False
    bar_format = f"{{l_bar}}{{bar:{bar_length}}}{{r_bar}}{{bar:{-bar_length}b}}"

    if enum:
        return tqdm.tqdm(enumerate(iterable), total=len(iterable), position=position, disable=hide_bar, desc=text, bar_format=bar_format, leave=True)
    else:
        return tqdm.tqdm(iterable, total=len(iterable), position=position, disable=hide_bar, desc=text, bar_format=bar_format, leave=True)


class Simulation(object):

    def __init__(self, geometry, light_sources, det_channel=None):
        self.geometry = geometry
        self.light_sources = light_sources
        self.det_channel = det_channel
        self.photon_hits = {}
        self.gpu_geometry = None
        self.gpu_photons = None

    def run(self, nthreads_per_block=64, max_blocks=1024, reps=1, max_steps=10, timestep=10000, time_offset=0):
        self.gpu_geometry = GPUDetector(self.geometry)
        rng_states = get_rng_states(nthreads_per_block*max_blocks)
        for rep in tqdm_range(start=0, stop=reps):
            if isinstance(self.light_sources, list):
                light_source_0 = self.light_sources[0]
                if len(self.light_sources) > 1:
                    light_source_0.join(self.light_sources[1:])
                self.gpu_photons = GPUPhotons(light_source_0)
            else:
                self.gpu_photons = GPUPhotons(self.light_sources)

            self.gpu_photons.propagate(self.gpu_geometry, rng_states, nthreads_per_block, max_blocks, max_steps)
            if self.det_channel is not None:
                hit_photons = self.gpu_photons.get_hits(gpu_detector=self.gpu_geometry)
                for det, channel in self.det_channel.items():
                    if det not in self.photon_hits.keys():
                        self.photon_hits[det] = hit_photons[channel][hit_photons[channel].channel == channel]
                        self.photon_hits[det].t += time_offset
                    else:
                        hit_photons[channel].t = hit_photons[channel].t + rep * timestep + time_offset
                        self.photon_hits[det] = self.photon_hits[det] + hit_photons[channel][hit_photons[channel].channel == channel]


class SimProcessor(object):

    def __init__(self, simulation, waveform_window, dt=2):
        self.simulation = simulation

        self.proc_list = []
        self.outputs = {}
        self.save_to_file = []
        self.settings = {}

    def process(self):
        for processor in self.proc_list:
            if isinstance(processor, SimProcessorBase):
                processor.process_block(self.outputs)
            else:
                raise TypeError("Couldn't identify processor type!")
        return {key: self.outputs[key] for key in self.save_to_file}

    def add(self, fun_name, settings):
        if settings is None:
            settings = {}
        if fun_name in self.settings:
            self.settings[fun_name] = {**self.settings[fun_name], **settings}
        else:
            self.settings[fun_name] = settings
        if fun_name in dir(func_dir):
            self.proc_list.append(
                SimProcessorBase(getattr(func_dir, fun_name), **self.settings[fun_name]))
        else:
            raise LookupError(f"Unknown function: {fun_name}")

    def init_outputs(self, outputs):
        self.outputs = outputs

    def add_output(self, key, value):
        self.outputs[key] = value

    def reset_outputs(self):
        self.outputs.clear()

    def add_to_file(self, var_name):
        if isinstance(var_name, str):
            if var_name not in self.save_to_file:
                self.save_to_file.append(var_name)
        elif isinstance(var_name, list):
            self.save_to_file += var_name
        else:
            raise TypeError(f"var_name of type {type(var_name)} must be str or list of strings")

    def clear(self):
        self.proc_list.clear()
        self.settings.clear()


class SimProcessorBase(object):
    def __init__(self, function, **kwargs):
        self.function = function
        self.fun_kwargs = kwargs

    def process_block(self, outputs):
        self.function(outputs, **self.fun_kwargs)


def load_functions(proc_settings, processor):
    for key, params in proc_settings["processes"].items():
        processor.add(key, settings=params)
    for output in proc_settings["save_output"]:
        processor.add_to_file(output)
    for output in proc_settings["save_waveforms"]:
        processor.add_to_file(output)


def process_data(input_path, output_path, processor, channels, bias=None, overwrite=False, verbose=False, chunk=2000, write_size=1):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_files = glob.glob(f"{input_path}/*.h5")
    data_files.sort()
    output_files = []
    for file in data_files:
        if bias is None:
            output_files.append(file.replace("raw", "t2"))
        else:
            if str(bias) in file:
                output_files.append(file.replace("raw", "t2"))

    if verbose:
        print(" ")
        print("Starting Chroma Sim processing ... ")
        print("Input Path: ", input_path)
        print("Output Path: ", output_path)
        print("Input Files: ", data_files)

        file_sizes = []
        for file_name in data_files:
            memory_size = os.path.getsize(file_name)
            memory_size = round(memory_size/1e6)
            file_sizes.append(str(memory_size)+" MB")
        print("File Sizes: ", file_sizes)

    if overwrite is True:
        for file_name in output_files:
            destination = os.path.join(file_name)
            if os.path.isfile(destination):
                os.remove(destination)

    start = time.time()
    # -----Processing Begins Here!---------------------------------

    for idx, file in enumerate(data_files):
        destination = os.path.join(input_path, file)
        output_destination = os.path.join(output_path, output_files[idx])
        if verbose:
            print(f"Processing: {file}")
        h5_file = h5py.File(destination, "r")
        h5_output_file = h5py.File(output_destination, "a")
        num_rows = h5_file["n_events"][()]
        data_storage = {"size": 0}
        for i in tqdm_range(0, num_rows//chunk + 1, verbose=verbose):
            begin, end = _chunk_range(i, chunk, num_rows)
            _initialize_outputs(h5_file, processor, begin, end)
            output_data = processor.process()
            _output_chunk(h5_output_file, output_data, data_storage, write_size, num_rows, chunk, end)
            processor.reset_outputs()
        _copy_to_t2(h5_file, h5_output_file)
        _output_date(output_destination, "process_date")
        h5_file.close()
        h5_output_file.close()

    if verbose:
        print("Processing Finished! ...")
        print("Output Files: ", [file.replace("raw", "t2") for file in data_files])
        _output_time(time.time() - start)


def _chunk_range(index, chunk, num_rows):
    start = index * chunk
    stop = (index+1) * chunk
    if stop >= num_rows:
        stop = num_rows
    return start, stop


def _initialize_outputs(h5_file, processor, begin, end):
    data_dict = {}
    for channel in h5_file["/raw/channels"].keys():
        data_dict["timetag"] = h5_file["timetag"][begin: end]
        data_dict[f"/raw/channels/{channel}/waveforms"] = h5_file[f"/raw/channels/{channel}/waveforms"][begin: end]
    processor.init_outputs(data_dict)


def _output_chunk(output_file, chunk_data, storage, write_size, num_rows, chunk, stop):
    output_to_file = False
    if (write_size == 1) | (num_rows < chunk):
        output_to_file = True
    elif stop >= num_rows-1:
        output_to_file = True
    elif storage["size"] == (write_size - 1):
        output_to_file = True

    for i, output in enumerate(chunk_data.keys()):
        if output not in storage:
            storage[output] = []
        storage[output].append(chunk_data[output])
        if i == 0:
            storage["size"] = len(storage[output])
        if output_to_file:
            storage[output] = np.concatenate(storage[output])
    if output_to_file:
        _output_to_file(output_file, storage)
        storage.clear()
        storage["size"] = 0


def _copy_to_t2(h5_file, output_file):
    for key in h5_file.keys():
        if key != "raw":
            output_file.create_dataset(key, data=h5_file[key])
    for channel in h5_file["/raw/channels"].keys():
        for key in h5_file[f"/raw/channels/{channel}"]:
            if key != "waveforms":
                output_file.create_dataset(f"/processed/channels/{channel}/{key}", data=h5_file[f"/raw/channels/{channel}/{key}"])


def _output_to_file(output_file, storage):
    for key, data in storage.items():
        if key == "size": continue
        if key in output_file:
            output_file[key].resize(output_file[key].shape[0]+data.shape[0], axis=0)
            output_file[key][-data.shape[0]:] = data
        else:
            if len(data.shape) == 2:
                output_file.create_dataset(key, data=data, maxshape=(None, None))
            elif len(data.shape) == 1:
                output_file.create_dataset(key, data=data, maxshape = (None,))
            else:
                raise ValueError(f"Dimension of output data {data.shape} must be 1 or 2")


def _output_date(output_destination, label=None):
    with h5py.File(output_destination, "a") as output_file:
        if label is None:
            label = "date"
        if label not in output_file.keys():
            output_file.create_dataset(label, data=int(time.time()))
        else:
            output_file[label] = int(time.time())


def _output_time(delta_seconds):
    temp_seconds = delta_seconds
    hours = 0
    minutes = 0

    while temp_seconds >= 3600:
        temp_seconds = temp_seconds - 3600
        hours = hours + 1

    while temp_seconds >= 60:
        temp_seconds = temp_seconds - 60
        minutes = minutes + 1
    seconds = round(temp_seconds, 1)
    print(" ")
    print(f"Time elapsed {hours}h {minutes}m {seconds}s")
    print(" ")
