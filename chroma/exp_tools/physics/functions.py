import numpy as np
import peakutils

from scipy.optimize import curve_fit
from functools import partial


def gaussian(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))


def baseline_subtract(outputs, wf_in, wf_out, degree=1, flip=False):
    waves_data = outputs[wf_in]
    if flip:
        waves_data = -outputs[wf_in]
    else:
        waves_data = outputs[wf_in]
    axis_function = partial(peakutils.baseline, deg=degree)
    baselines = np.apply_along_axis(axis_function, 1, waves_data)
    outputs[wf_out] = waves_data - baselines


def baseline_subtract_simple(outputs, wf_in, wf_out, t_range=[0, 100], flip=False):
    waves_data = outputs[wf_in]
    if flip:
        waves_data = -outputs[wf_in]
    def axis_function(waveform):
        first_average = np.mean(waveform[t_range[0]: t_range[1]])
        baseline = np.repeat(first_average, repeats=len(waveform))
        return baseline
    baselines = np.apply_along_axis(axis_function, 1, waves_data)
    outputs[wf_out] = waves_data - baselines


def baseline_subtract_gauss(outputs, wf_in, wf_out, sample_range=None, flip=False):
    waves_data = outputs[wf_in]
    if flip:
        waves_data = -outputs[wf_in]
    def axis_function(waveform):
        fit_waveform = waveform
        if sample_range is not None:
            fit_waveform = waveform[sample_range[0]: sample_range[1]]
        n, bins = np.histogram(fit_waveform, bins=100)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        max_loc = np.where(n == max(n))[0][0]
        coeffs, covs = curve_fit(gaussian, bin_centers, n, p0=[bin_centers[max_loc], 100, 100])
        base_value = coeffs[0]
        return np.repeat(base_value, repeats=len(waveform))
    try:
        baselines = np.apply_along_axis(axis_function, 1, waves_data)
        outputs[wf_out] = waves_data - baselines
    except:
        if sample_range is None:
            baseline_subtract_simple(outputs, wf_in, wf_out)
        else:
            baseline_subtract_simple(outputs, wf_in, wf_out, t_range=sample_range, flip=flip)


def charge(outputs, wf_in, out, window):
    waveforms = outputs[wf_in]
    charges = integrate_current(waveforms, window[0], window[1])
    outputs[out] = charges


def normalize_charge(outputs, in_name, out_name, peak_locs, peak_errors):
    x0 = peak_locs[0]
    peak_locs = np.array(peak_locs)[1:]
    peak_errors = np.array(peak_errors)[1:]
    peak_diffs = peak_locs[1:] - peak_locs[:-1]
    diff_errors = np.sqrt((peak_errors[1:] + peak_errors[:-1])**2)
    gain = np.sum(peak_diffs * diff_errors) / np.sum(diff_errors)
    charges = outputs[in_name]
    outputs[out_name] = (charges - x0) / gain


def integrate_current(current_forms, lower_bound=0, upper_bound=200, sample_time=2e-9):
    return np.sum(current_forms.T[lower_bound:upper_bound].T, axis=1)*sample_time