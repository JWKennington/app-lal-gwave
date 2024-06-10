"""This module provides a simplified interface to CBC waveforms using LALSuite.
"""
from typing import Tuple

import lal
import lalsimulation
import numpy
import pandas
import xarray
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows

APPROXIMANTS = [
    'IMRPhenomD',
    'IMRPhenomPv2',
]

SAMPLE_RATE = 512.0
DATA_DURATION = 20.0


def _waveform_base_kwargs(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD') -> dict:
    """Helper func for base kwargs"""
    kwargs = {
        # Component masses
        'm1': m1,
        'm2': m2,

        # First component Spin Vector
        'S1x': 0.,
        'S1y': 0.,
        'S1z': s1z,

        # Second component spin vector
        'S2x': 0.,
        'S2y': 0.,
        'S2z': s2z,

        # Approximant to use
        'approximant': lalsimulation.GetApproximantFromString(approximant),

        # Frequency domain parameters
        'f_min': 15.0,
        'f_ref': 0.0,

        # Standard Extrinsic Parameters
        'distance': 1.e6 * lal.PC_SI,
        'inclination': 0.,
        'phiRef': 0.,
        'longAscNodes': 0.,
        'eccentricity': 0.,
        'meanPerAno': 0.,

        # Other parameters
        'LALparams': None,
    }

    kwargs['m1'] *= lal.MSUN_SI
    kwargs['m2'] *= lal.MSUN_SI

    return kwargs


def _waveform_td(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD') -> Tuple[lal.COMPLEX16FrequencySeries, ...]:
    # Get waveform kwargs
    kwargs = _waveform_base_kwargs(m1, m2, s1z, s2z, approximant)

    # Make time-domain kwargs
    kwargs_td = kwargs.copy()
    kwargs_td['deltaT'] = 1.0 / SAMPLE_RATE

    return lalsimulation.SimInspiralTD(**kwargs_td)


def _waveform_fd(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD') -> Tuple[lal.COMPLEX16FrequencySeries, ...]:
    # Get waveform kwargs
    kwargs = _waveform_base_kwargs(m1, m2, s1z, s2z, approximant)

    # Make frequency-domain kwargs
    kwargs_fd = kwargs.copy()
    kwargs_fd['f_max'] = SAMPLE_RATE
    kwargs_fd['deltaF'] = 1.0 / 64.0

    return lalsimulation.SimInspiralFD(**kwargs_fd)


def _waveform_inner_product(w1: numpy.ndarray, w2: numpy.ndarray) -> float:
    return numpy.abs((numpy.conj(w1) * w2).sum())


def _waveform_norm(w: numpy.ndarray) -> float:
    """Compute a trapezoidal norm

    Args:
        w:
            Array, the array for which to compute the norm

    Returns:
        float, the norm
    """
    return _waveform_inner_product(w, w) ** 0.5


def _waveform_match(w1: lal.COMPLEX16FrequencySeries, w2: lal.COMPLEX16FrequencySeries, dt: float = 0.0) -> float:
    x = numpy.copy(w1.data.data)
    y = numpy.copy(w2.data.data)
    freq_vec = numpy.arange(len(x)) * w1.deltaF + w1.f0
    if w1.epoch != w2.epoch or dt:
        y *= numpy.exp(-2.j * numpy.pi * freq_vec * (dt + float(w2.epoch - w1.epoch)))
    y /= _waveform_norm(y)
    x /= _waveform_norm(x)
    m = _waveform_inner_product(x, y)
    return m


def _match_at_coords(m1: float, m2: float, s1z: float, s2z: float, guess_m1: float, guess_m2: float, guess_s1z: float, guess_s2z: float, approximant: str = 'IMRPhenomD', polarization: str = 'plus') -> float:
    """Get the mismatch between the guess and the true waveform."""
    w1 = _waveform_fd(m1, m2, s1z, s2z, approximant)[0 if polarization == 'plus' else 1]
    w2 = _waveform_fd(guess_m1, guess_m2, guess_s1z, guess_s2z, approximant)[0 if polarization == 'plus' else 1]

    return _waveform_match(w1, w2)


def _gen_data(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD', duration: float = DATA_DURATION, t0: float = None, polarization: str = 'plus') -> numpy.ndarray:
    """Generate the CBC waveform data."""
    # Generate array of zeros of length duration at sample rate
    ts = numpy.arange(0, duration, 1 / SAMPLE_RATE)
    data = numpy.zeros(shape=(len(ts),))

    # Generate waveform data
    hplus_td, hcross_td = _waveform_td(m1, m2, s1z, s2z, approximant)
    strain = hplus_td.data.data if polarization == 'plus' else hcross_td.data.data

    # Find start time array index
    if t0 is None:
        mid_idx = len(ts) // 2
        t0_idx = mid_idx - len(strain) // 2
    else:
        t0_idx = int(t0 * SAMPLE_RATE)

    # Inject the waveform data
    data[t0_idx:t0_idx + len(strain)] = strain

    # Return data
    return data


def _gen_noise(duration: float = DATA_DURATION) -> numpy.ndarray:
    """Generate the noise data."""
    # Generate array of zeros of length duration at sample rate
    ts = numpy.arange(0, duration, 1 / SAMPLE_RATE)
    data = numpy.random.randn(len(ts))

    # Return data
    return data


def get_cbc_waveform(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD', include_freq: bool = True) -> pandas.DataFrame:
    """Get a CBC waveform for a given binary system.

    Args:
        m1:
            float, mass of the first component in solar masses
        m2:
            float, mass of the second component in solar masses
        s1z:
            float, dimensionless spin of the first component
        s2z:
            float, dimensionless spin of the second component

    Returns:
        pandas.DataFrame:
            A DataFrame with columns 'time', 'strain', and 'polarization'.
    """
    # Get waveform data
    hplus_td, hcross_td = _waveform_td(m1, m2, s1z, s2z, approximant)
    ts = numpy.arange(0, len(hplus_td.data.data) * hplus_td.deltaT, hplus_td.deltaT)
    data = pandas.concat([
        pandas.DataFrame({'x': ts, 'strain': hplus_td.data.data}).assign(polarization='plus', domain='time', component='real'),
        pandas.DataFrame({'x': ts, 'strain': hcross_td.data.data}).assign(polarization='cross', domain='time', component='real'),
    ], axis=0)

    if include_freq:
        # Get frequency domain data
        hplus_fd, hcross_fd = _waveform_fd(m1, m2, s1z, s2z, approximant)
        fs = numpy.arange(0, len(hplus_fd.data.data) * hplus_fd.deltaF, hplus_fd.deltaF)
        data_fd = pandas.concat([
            pandas.DataFrame({'x': fs, 'strain': numpy.real(hplus_fd.data.data)}).assign(polarization='plus', domain='freq', component='real'),
            pandas.DataFrame({'x': fs, 'strain': numpy.imag(hplus_fd.data.data)}).assign(polarization='plus', domain='freq', component='imag'),
            pandas.DataFrame({'x': fs, 'strain': numpy.real(hcross_fd.data.data)}).assign(polarization='cross', domain='freq', component='real'),
            pandas.DataFrame({'x': fs, 'strain': numpy.imag(hcross_fd.data.data)}).assign(polarization='cross', domain='freq', component='imag'),
        ], axis=0)

        # Concatenate the data
        data = pandas.concat([data, data_fd], axis=0)

    return data


def get_spectrogram_data(data: pandas.DataFrame, polarization: str = 'plus', win_m: int = 128, win_std: int = 16) -> xarray.DataArray:
    """Get the spectrogram data for a given CBC waveform.

    Args:
        data:
            pandas.DataFrame, the waveform data

    Returns:
        pandas.DataFrame:
            A DataFrame with columns 'x', 'y', and 'z'.
    """
    # Filter for time domain
    data = data[(data['domain'] == 'time') & (data['polarization'] == polarization)]
    ts = data['x'].values
    ys = data['strain'].values

    # Get the spectrogram using ShortTimeFFT spectrogram
    win = windows.gaussian(M=win_m, std=win_std)
    sft = ShortTimeFFT(win=win, hop=2, fs=1 / (ts[1] - ts[0]), scale_to='psd')
    Sxx = sft.spectrogram(ys)

    # Reshape the data
    Sxx = numpy.log10(Sxx)
    Sxx = numpy.clip(Sxx, numpy.percentile(Sxx, 5), numpy.percentile(Sxx, 95))

    # Compute time coordinates
    ts = numpy.arange(0, Sxx.shape[1]) * sft.hop / sft.fs

    # Make the data with time and frequency coordinates
    data = xarray.DataArray(Sxx, dims=['frequency', 'time'], coords={'time': ts, 'frequency': sft.f})

    return data


def generate_audio_file(m1: float, m2: float, s1z: float, s2z: float, approximant: str = 'IMRPhenomD', polarization: str = 'plus', return_filepath: bool = False, shift_freq: bool = False):
    # Format unique filename for output .wav file
    file = f'assets/{m1:.1f}_{m2:.1f}_{s1z:.1f}_{s2z:.1f}_{approximant}_{polarization}.wav'

    # Get waveform data (time_domain only)
    data = get_cbc_waveform(m1, m2, s1z, s2z, approximant, include_freq=False)

    return generate_audio_file_from_data(file, data, polarization, return_filepath, shift_freq)


def generate_audio_file_from_data(file: str, data: pandas.DataFrame, polarization: str = 'plus', return_filepath: bool = False, shift_freq: bool = False):
    # Extract amplitude array
    ys = data[(data['polarization'] == polarization) & (data['domain'] == 'time')]['strain'].values

    # Normalize amplitude array
    ys = ys / numpy.max(numpy.abs(ys))

    # Scale amplitude array to reasonable volume
    ys = ys * 2.0

    # Shift frequency up by 400Hz if requested
    if shift_freq:
        # Compute FFT using scipy
        Ys = numpy.fft.fft(ys)
        fs = numpy.fft.fftfreq(len(ys), 1 / SAMPLE_RATE)

        # Shift frequency, determine shift of 400 Hz in term of array roll
        shift = int(200 // (fs[1] - fs[0]))
        Ys = numpy.roll(Ys, shift)
        # mute wrapped components
        Ys[:shift] = 0

        # Compute inverse FFT
        ys = numpy.fft.ifft(Ys)
        ys = numpy.real(ys)

    # Write audio file
    wavfile.write(file, int(SAMPLE_RATE), ys)

    if return_filepath:
        return file


def get_mismatch_guess(m1: float, m2: float, s1z: float, s2z: float, approximant: str, polarization: str, guess_m1: float, guess_m2: float, guess_s1z: float, guess_s2z: float) -> float:
    """Get the mismatch between the guess and the true waveform."""
    data_true = get_cbc_waveform(m1, m2, s1z, s2z, approximant)
    data_guess = get_cbc_waveform(guess_m1, guess_m2, guess_s1z, guess_s2z, approximant)

    data_true = data_true[(data_true['polarization'] == polarization) & (data_true['domain'] == 'time')]
    data_guess = data_guess[(data_guess['polarization'] == polarization) & (data_guess['domain'] == 'time')]

    ts = data_true['x'].values
    ys_true = data_true['strain'].values
    ys_guess = data_guess['strain'].values

    # Compute the mismatch
    mismatch = numpy.sum((ys_true - ys_guess) ** 2)

    return mismatch


def get_fake_data(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD', duration: float = 60, noise_scale: float = 1.0, polarization: str = 'plus',
                  signal_scale: float = 1.0) -> pandas.DataFrame:
    """Get fake data for testing."""
    # Generate waveform data
    data_event = _gen_data(m1, m2, s1z, s2z, approximant, duration=duration, polarization=polarization)
    data_noise = _gen_noise(duration=duration)

    # Scale noise to ration of event amplitude
    data_noise = data_noise * (noise_scale * numpy.max(numpy.abs(data_event)) / numpy.max(numpy.abs(data_noise)))

    # Inject event into noise
    data = signal_scale * data_event + data_noise

    # Convert to DataFrame
    ts = numpy.arange(0, duration, 1 / SAMPLE_RATE)
    data = pandas.DataFrame({'x': ts, 'strain': data}).assign(polarization='plus',
                                                              domain='time',
                                                              component='real')
    return data


def get_snr_data(m1: float, m2: float, s1z: float, s2z: float, data: pandas.DataFrame) -> pandas.DataFrame:
    """Get the SNR data for a given CBC waveform."""
    # Get waveform template data
    hplus_td, hcross_td = _waveform_td(m1, m2, s1z, s2z)
    template = hplus_td.data.data

    # Normalize template
    template = template / _waveform_norm(template)

    # Extract detector data
    ts = data['x'].values
    ys = data['strain'].values

    # Compute signal amplitude using time convolution of template array through strain array
    sigamp = numpy.convolve(ys, template[::-1], mode='valid')
    sigamp_ts = ts[len(ts) - len(sigamp):]

    # Extend sigamp array to match length of strain array padding at the beginning of the array
    # sigamp = numpy.pad(sigamp, (len(ts) - len(sigamp), 0))

    # Compute denominator from purely noisy part
    noise_idx = len(ys) // 3
    noise = ys[:noise_idx]
    noise_var = numpy.mean(noise ** 2)


    # Compute SNR
    snr = sigamp ** 2 / noise_var

    # Extend snr array to match length of strain array padding at the beginning and end
    pad_front = (len(ts) - len(sigamp)) // 2
    pad_end = len(ts) - len(sigamp) - pad_front
    snr = numpy.pad(snr, (pad_front + pad_end, 0), constant_values=(snr[0], snr[-1]))

    # Package as dataframe
    data = pandas.DataFrame({'x': ts, 'y': snr})

    return data

