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

    def run(self, rng_states, nthreads_per_block=64, max_blocks=1024, reps=1, max_steps=10):
        for rep in range(reps):
            self.gpu_geometry = GPUDetector(self.geometry)
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
                        self.photon_hits[det] = self.photon_hits[det] + hit_photons[hit_photons.channel == channel]
        