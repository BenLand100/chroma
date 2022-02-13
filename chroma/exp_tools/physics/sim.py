from chroma.gpu.geometry import GPUGeometry
from chroma.gpu.detector import GPUDetector
from chroma.gpu.photon import GPUPhotons

class Simulation(object):

    def __init__(self, geometry, light_sources):
        self.geometry = geometry
        self.light_sources = light_sources
        self.gpu_geometry = None
        self.gpu_photons = None

    def run(self, rng_states, nthreads_per_block=64, max_blocks=1024, reps=1, max_steps=10):
        self.gpu_geometry = GPUDetector(self.geometry)
        if isinstance(self.light_sources, list):
            light_source_0 = self.light_sources[0]
            if len(self.light_sources) > 1:
                light_source_0.join(self.light_sources[1:])
            self.gpu_photons = GPUPhotons(light_source_0)
        else:
            self.gpu_photons = GPUPhotons(self.light_sources)

        self.gpu_photons.propagate(self.gpu_geometry, rng_states, nthreads_per_block, max_blocks, max_steps)
        