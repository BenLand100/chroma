import pandas as pd

from chroma.gpu.geometry import GPUGeometry
from chroma.gpu.detector import GPUDetector
from chroma.gpu.photon import GPUPhotons

class Simulation(object):

    def __init__(self, geometry, light_sources, det_channel=None):
        self.geometry = geometry
        self.light_sources = light_sources
        self.det_channel = det_channel
        self.photon_hits = {}
        self.gpu_geometry = None
        self.gpu_photons = None

    def run(self, rng_states, nthreads_per_block=64, max_blocks=1024, reps=1, max_steps=10, timestep=10000):
        self.gpu_geometry = GPUDetector(self.geometry)
        for rep in range(reps):
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
                        self.photon_hits[det] = hit_photons[hit_photons.channel == channel]
                    else:
                        hit_photons.t = hit_photons.t + timestep
                        self.photon_hits[det] = self.photon_hits[det] + hit_photons[hit_photons.channel == channel]


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
        if fun_name in dir(pc):
            self.proc_list.append(
                SimProcessorBase(getattr(pc, fun_name), **self.settings[fun_name]))
        elif fun_name in dir(pt):
            self.proc_list.append(
                SimProcessorBase(getattr(pt, fun_name), **self.settings[fun_name]))
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