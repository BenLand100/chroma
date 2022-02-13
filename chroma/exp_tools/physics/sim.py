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


def sipm_params(datasheet, bias=None):
    if bias is None:
        bias = 28

    sipm_data = pd.read_csv(datasheet)
    biases = list(sipm_data["bias"])
    idx = biases.index(bias)
    result = [
        sipm_data["gain"][idx],
        sipm_data["dark_rate"][idx],
        sipm_data["cross_talk"][idx],
        sipm_data["afterpulse"][idx]
    ]
    return result
        