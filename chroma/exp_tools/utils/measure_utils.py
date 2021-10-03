import tqdm
import numpy as np
from chroma.event import Photons


class PlaneBoundary(object):

    def __init__(self, center, normal):
        self.center = center
        self.normal = normal

    def compute_crossings(self, step_photons, step_photon_ids, distance):
        photon_starts = []
        photon_stops = []
        all_starts = None
        all_stops = None
        for i in tqdm.tqdm(range(len(step_photons)-1)):
            start_ids = step_photon_ids[i]
            stop_ids = step_photon_ids[i+1]
            common, idx_start, idx_stop = np.intersect1d(start_ids, stop_ids, return_indices=True)
            photons_init = step_photons[i][idx_start]
            photons_final = step_photons[i+1][idx_stop]

            slopes = photons_final.pos - photons_init.pos
            direction = np.sign(np.sum(np.multiply(slopes, self.normal), axis=1))

            slopes = slopes[direction == -1]
            photons_init = photons_init[direction == -1]
            photons_final = photons_final[direction == -1]

            diff = -np.subtract(photons_init.pos, self.center.T)
            t_vector = np.sum(np.multiply(diff, self.normal), axis=1) / np.sum(np.multiply(slopes, self.normal), axis=1)
            steps = np.multiply(slopes.T, t_vector)
            intersect_points = photons_init.pos + steps.T

            in_range = np.sum(np.multiply(slopes, intersect_points - photons_final.pos), axis=1)

            distance_squared = np.sum(np.subtract(intersect_points, self.center.T) * np.subtract(intersect_points, self.center.T), axis=1)
            pass_through = (distance_squared < distance**2) & (t_vector > 0) & (in_range < 0)
            if i == 0:
                all_starts = photons_init[pass_through]
                all_stops = photons_init[pass_through]
            else:
                photon_starts.append(photons_init[pass_through])
                photon_stops.append(photons_final[pass_through])

        return all_starts.join(photon_starts), all_stops.join(photon_stops)