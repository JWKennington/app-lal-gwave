"""Plotting utilities
"""

import pandas
import xarray
from plotly import express


def cbc_time_domain(data: pandas.DataFrame, polarization='plus'):
    """Plot a CBC waveform for a given binary system.

    Args:
        m1:
            float, mass of the first component in solar masses
        m2:
            float, mass of the second component in solar masses
        s1z:
            float, dimensionless spin of the first component
        s2z:
            float, dimensionless spin of the second component
    """
    # Filter for time domain
    data = data[data['domain'] == 'time']

    if polarization != 'both':
        data = data[data['polarization'] == polarization]

    fig = express.line(data, x='x', y='strain', color='polarization', title='Time Domain')

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Strain')

    return fig


def cbc_freq_domain(data: pandas.DataFrame, polarization='plus'):
    """Plot a CBC waveform for a given binary system.

    Args:
        m1:
            float, mass of the first component in solar masses
        m2:
            float, mass of the second component in solar masses
        s1z:
            float, dimensionless spin of the first component
        s2z:
            float, dimensionless spin of the second component
    """
    # Filter for frequency domain
    data = data[(data['domain'] == 'freq') & (data['component'] == 'real')]

    if polarization != 'both':
        data = data[data['polarization'] == polarization]

    fig = express.line(data, x='x', y='strain', color='polarization', title='Frequency Domain')

    # Set axis labels
    fig.update_xaxes(title_text='Frequency (Hz)')
    fig.update_yaxes(title_text='Amplitude')

    return fig


def cbc_spectrogram(data: xarray.DataArray):
    """Plot a spectrogram of a CBC waveform for a given binary system.

    Args:
        data:
            xarray.DataArray, a spectrogram of a CBC waveform
    """
    fig = express.imshow(data, title='Spectrogram', origin='lower')

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Frequency (Hz)')

    return fig

