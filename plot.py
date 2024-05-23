"""Plotting utilities
"""

import pandas
import xarray
from plotly import express

HEIGHT_LINE = 250
HEIGHT_IMSHOW = 2 * HEIGHT_LINE
WIDTH_LINE = 700
WIDTH_IMSHOW = 0.8 * WIDTH_LINE
PLOT_MARGIN = dict(l=0, r=0, t=10, b=10)

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

    fig = express.line(data, x='x', y='strain', color='polarization',
                       height=HEIGHT_LINE,
                       width=WIDTH_LINE)

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Strain')

    # Set minimal margin
    fig.update_layout(margin=PLOT_MARGIN)

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

    fig = express.line(data, x='x', y='strain', color='polarization',
                       height=HEIGHT_LINE,
                       width=WIDTH_LINE)

    # Set axis labels
    fig.update_xaxes(title_text='Frequency (Hz)')
    fig.update_yaxes(title_text='Amplitude')

    # Set minimal margin
    fig.update_layout(margin=PLOT_MARGIN)

    return fig


def cbc_spectrogram(data: xarray.DataArray):
    """Plot a spectrogram of a CBC waveform for a given binary system.

    Args:
        data:
            xarray.DataArray, a spectrogram of a CBC waveform
    """
    fig = express.imshow(data, origin='lower',
                         height=HEIGHT_IMSHOW,
                         width=WIDTH_IMSHOW)

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Frequency (Hz)')

    # Set minimal margin
    fig.update_layout(margin=PLOT_MARGIN)

    return fig
