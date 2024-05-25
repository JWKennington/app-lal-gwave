"""This module provides a simplified interface to CBC waveforms using LALSuite.
"""

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

SAMPLE_RATE = 256.0


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

    # Make time-domain kwargs
    kwargs_td = kwargs.copy()
    kwargs_td['deltaT'] = 1.0 / SAMPLE_RATE

    # Make frequency-domain kwargs
    kwargs_fd = kwargs.copy()
    kwargs_fd['f_max'] = 256.0
    kwargs_fd['deltaF'] = 1.0 / 64

    hplus_td, hcross_td = lalsimulation.SimInspiralTD(**kwargs_td)

    # Make array of time coordinates
    ts = numpy.arange(0, len(hplus_td.data.data) * hplus_td.deltaT, hplus_td.deltaT)

    data = pandas.concat([
        pandas.DataFrame({'x': ts, 'strain': hplus_td.data.data}).assign(polarization='plus', domain='time', component='real'),
        pandas.DataFrame({'x': ts, 'strain': hcross_td.data.data}).assign(polarization='cross', domain='time', component='real'),
    ], axis=0)

    if include_freq:
        hplus_fd, hcross_fd = lalsimulation.SimInspiralFD(**kwargs_fd)

        # Make array of frequency coordinates
        fs = numpy.arange(0, len(hplus_fd.data.data) * hplus_fd.deltaF, hplus_fd.deltaF)

        data_fd = pandas.concat([
            pandas.DataFrame({'x': fs, 'strain': numpy.real(hplus_fd.data.data)}).assign(polarization='plus', domain='freq', component='real'),
            pandas.DataFrame({'x': fs, 'strain': numpy.imag(hplus_fd.data.data)}).assign(polarization='plus', domain='freq', component='imag'),
            pandas.DataFrame({'x': fs, 'strain': numpy.real(hcross_fd.data.data)}).assign(polarization='cross', domain='freq', component='real'),
            pandas.DataFrame({'x': fs, 'strain': numpy.imag(hcross_fd.data.data)}).assign(polarization='cross', domain='freq', component='imag'),
        ], axis=0)

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


def generate_audio_file(m1: float, m2: float, s1z: float, s2z: float, approximant: str = 'IMRPhenomD', polarization: str = 'plus', return_filepath: bool = False):
    # Format unique filename for output .wav file
    file = f'assets/{m1:.1f}_{m2:.1f}_{s1z:.1f}_{s2z:.1f}_{approximant}_{polarization}.wav'

    # Get waveform data (time_domain only)
    data = get_cbc_waveform(m1, m2, s1z, s2z, approximant, include_freq=False)

    # Extract amplitude array
    ys = data[(data['polarization'] == polarization) & (data['domain'] == 'time')]['strain'].values

    # Normalize amplitude array
    ys = ys / numpy.max(numpy.abs(ys))

    # Write audio file
    wavfile.write(file, int(SAMPLE_RATE), ys)

    if return_filepath:
        return file


def get_fake_data(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD', duration: float = 60) -> pandas.DataFrame:
    """Get fake data for testing."""
    ts = numpy.linspace(0, 1, 1000)
    ys = numpy.sin(2 * numpy.pi * 10 * ts)
    data = pandas.DataFrame({'x': ts, 'strain': ys, 'polarization': 'plus', 'domain': 'time', 'component': 'real'})
    return data
