import numpy as np


def rgb_surface(rgb_array):
    inverse_array = np.array([255, 255, 255]) - np.array(rgb_array)
    norm_rgb = np.array(rgb_array) / 255
    norm_rgb_inverse = inverse_array / 255
    wavelengths = np.array([60, 480, 550, 580, 1000])
    reflectance = np.concatenate(([0], norm_rgb, [0]))
    absorb = np.concatenate(([1], norm_rgb_inverse, [1]))
    return reflectance, absorb, wavelengths
