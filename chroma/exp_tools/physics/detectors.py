from abc import ABC, abstractmethod

import math
from tkinter import Scale
import h5py
import time
import numpy as np
import scipy.constants as const

def poisson_distro(k, lam):
    if isinstance(k, int):
        return lam**k*np.exp(-lam) / math.factorial(k)
    elif (isinstance(k, list)) or (isinstance(k, np.ndarray)):
        output = []
        for sample in k:
            output.append(lam**sample*np.exp(-lam) / math.factorial(sample))
        return output
    else:
        raise AttributeError("k must be an int or list of ints!")


def borel_distro(mu, n):
    if isinstance(n, int):
        return np.exp(-mu*n) * (mu*n)**(n-1) / math.factorial(n)
    elif (isinstance(n, list)) or (isinstance(n, np.ndarray)):
        output = []
        for sample in n:
            output.append(np.exp(-mu*sample) * (mu*sample)**(sample-1) / math.factorial(sample))
        return output
    else:
        raise AttributeError("n must be an int or list of ints!")


def cross_talk_generator(prob):
    mu = prob / (prob + 1)
    choices = np.arange(1, 30, dtype=int)
    chances = borel_distro(mu, choices)
    return np.random.choice(choices, p=chances) - 1


def afterpulse_generator(prob, tau, decay_time):
    choices = [0, 1]
    chances = [1-prob, prob]
    choice = np.random.choice(choices, p=chances)
    if choice == 1:
        dt = np.random.exponential(scale=tau)
        amp = (1 - np.exp(-dt / decay_time))
        return amp, dt
    else:
        return None, None


def single_exp_response(t, t0, A, tau):
    t_before = t[t < t0]
    t_after = t[t >= t0]
    start = np.zeros(len(t_before))
    decay = (A/tau)*np.exp(-(t_after-t0)/tau)
    return np.concatenate((start, decay))


def double_exp_response(t, t0, A, amp_frac, short_tau, long_tau):
    t_before = t[t < t0]
    t_after = t[t >= t0]
    exp_short = (1 / (short_tau + amp_frac**-1*long_tau)) * np.exp(-(t_after - t0)/short_tau)
    exp_long = (1 / (short_tau*amp_frac + long_tau)) * np.exp(-(t_after-t0)/long_tau)
    return A * (exp_short + exp_long)


class DetectorDevice(ABC):

    def __init__(self, waveform_length, dt, hit_photons=None):
        self.hit_photons = hit_photons
        self.waveform_length = waveform_length
        self.dt = dt
        self.waves = None
        self.time = None

    def init_waves(self, num_waveforms):
        self.waves = np.zeros((num_waveforms, int(self.waveform_length / self.dt)), dtype=float)
        self.time = np.arange(0, int(self.waveform_length), self.dt)

    def generate_waveforms(self):
        pass

    def save_waveforms(self, output_path, channel_name=None, write_mode="w"):
        if channel_name is None:
            channel_name = "detector"
        with h5py.File(output_path, write_mode) as h5_file:
            h5_file.create_dataset(f"/raw/channels/{channel_name}/waveforms", data=self.waves)
            h5_file.create_dataset(f"/raw/channels/{channel_name}/wf_len", data=self.waves.shape[1])
            timetags = np.arange(0, self.waves.shape[0]*self.waveform_length, self.waveform_length)
            if "timetag" not in h5_file.keys():
                h5_file.create_dataset("timetag", data=timetags)
            if "dt" not in h5_file.keys():
                h5_file.create_dataset("dt", data=self.dt)
            if "date" not in h5_file.keys():
                h5_file.create_dataset("date", data=time.time())


class SiPM(DetectorDevice):

    def __init__(self, waveform_length, dt, hit_photons=None, dark_rate=800, cross_talk=0.3, afterpulse=1e-3):
        self.dark_rate = dark_rate
        self.cross_talk = cross_talk
        self.afterpulse = afterpulse
        self.all_times = []
        self.all_amps = []

        super(SiPM, self).__init__(waveform_length, dt, hit_photons)

    @staticmethod
    def __baseline_noise(t, fwhm_noise, fwhm_offset):
        noise = np.random.normal(loc=0, scale=fwhm_noise/(2*np.sqrt(2*np.log(2))), size=len(t))
        offset = np.random.normal(loc=0, scale=fwhm_offset/(2*np.sqrt(2*np.log(2))))
        return noise + offset

    def add_dark_counts(self, sample_size=100, trigger=None):
        for i, waveform in enumerate(self.waves):
            dark_tau = (1 / (self.dark_rate*1e3)) * 1e9
            dark_times = np.random.exponential(dark_tau, size=sample_size)
            if trigger is not None:
                dark_times = np.insert(dark_times, 0, trigger)
            time_locs = np.cumsum(dark_times)
            time_locs = time_locs[time_locs < self.time[-1]]
            amplitudes = np.ones(len(time_locs))
            for j, time in enumerate(time_locs):
                if amplitudes[j] >= 1:
                    dict_number = cross_talk_generator(self.cross_talk)
                    amplitudes[j] += dict_number
                    for k in range(dict_number+1):
                        amp, dt = afterpulse_generator(self.afterpulse, 100, 60)
                        if dt is not None:
                            amplitudes = np.append(amplitudes, amp)
                            time_locs = np.append(time_locs, time_locs[j]+dt)
            self.all_amps.append(amplitudes)
            self.all_times.append(time_locs)

    def add_light_counts(self):
        time_window = self.waveform_length * self.dt
        offset = 0
        all_hit_times = self.hit_photons.t
        for i, waveform in enumerate(self.waves.T):
            hit_times = all_hit_times[(all_hit_times >= offset) & (all_hit_times < offset + time_window)]
            amplitudes = np.ones(len(hit_times))
            offset += time_window
            for j, time in enumerate(hit_times):
                if amplitudes[j] >= 1:
                    dict_number = cross_talk_generator(self.cross_talk)
                    amplitudes[j] += dict_number
                    for k in range(dict_number+1):
                        amp, dt = afterpulse_generator(self.afterpulse, 100, 60)
                        if dt is not None:
                            amplitudes = np.append(amplitudes, amp)
                            time_locs = np.append(hit_times, hit_times[j]+dt)
                self.all_amps[j] = np.append(self.all_amps[j], amplitudes)
                self.all_times[j] = np.append(self.all_times[j], hit_times)

    def generate_waveforms(self, response_func, noise=True, fwhm_pe=0.25, fwhm_noise=0.1, fwhm_offset=0.05):
        for i, waveform in enumerate(self.waves):
            for j, time in enumerate(self.all_times[i]):
                amp = self.all_amps[i][j]
                if noise & (amp >= 1):
                    amp = np.random.normal(amp, scale=(fwhm_pe/(2*np.sqrt(2*np.log(2))))*np.sqrt(amp))
                elif noise & (amp < 1):
                    amp = np.random.normal(amp, scale=fwhm_pe/(2*np.sqrt(2*np.log(2))))
                self.waves[i] = np.add(self.waves[i], response_func(self.time, time, amp))
            if noise:
                self.waves[i] = np.add(self.waves[i], self.__baseline_noise(self.time, fwhm_noise, fwhm_offset))


class Photodiode(DetectorDevice):

    def __init__(self, waveform_length, dt, hit_photons=None):
        super(Photodiode, self).__init__(waveform_length, dt, hit_photons)

    @staticmethod
    def __baseline_noise(t, fwhm_noise, fwhm_offset):
        noise = np.random.normal(loc=0, scale=fwhm_noise/(2*np.sqrt(2*np.log(2))), size=len(t))
        offset = np.random.normal(loc=0, scale=fwhm_offset/(2*np.sqrt(2*np.log(2))))
        return noise + offset

    def generate_waveforms(self, response=None, noise=True, fwhm_noise=1e-20, fwhm_offset=0, scale=1e-14):
        if response is None:
            response = 23
        if self.hit_photons is None:
            raise ValueError("No hit_photons specified! Need hit_photons to calculate photocurrent!")
        hit_times = self.hit_photons.t
        hit_energies = const.h*const.c / (self.hit_photons.wavelengths*1e-9)
        t0 = 0
        tf = self.waveform_length
        for i, wave in enumerate(self.waves):
            if noise == True:
                self.waves[i] = self.__baseline_noise(self.time, fwhm_noise, fwhm_offset)
            local_hit_times = hit_times[(hit_times >= t0) & (hit_times < tf)]
            local_hit_energies = hit_energies[(hit_times >= t0) & (hit_times < tf)]
            n, bins_t = np.histogram(local_hit_times, bins=int(self.waveform_length/self.dt), 
                                     range=[0, int(self.waveform_length)], 
                                     weights=local_hit_energies)
            if len(local_hit_times) == 0:
                continue
            self.waves[i] += n * response * scale
            t0 += self.waveform_length
            tf += self.waveform_length