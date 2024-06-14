"""Plotting utilities
"""
import numpy
import pandas
import xarray
from plotly import express

HEIGHT_LINE = 250
HEIGHT_IMSHOW = 2 * HEIGHT_LINE
WIDTH_LINE = 700
WIDTH_IMSHOW = 0.8 * WIDTH_LINE
PLOT_MARGIN = dict(l=0, r=0, t=10, b=10)


def cbc_time_domain(data: pandas.DataFrame, polarization='plus', scale: bool = False):
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
    data = data.copy()

    # Filter for time domain
    data = data[data['domain'] == 'time']

    if polarization != 'both':
        data = data[data['polarization'] == polarization]

    y_label = 'Strain'

    if scale:
        # Determine power of 10 to scale the strain
        pow10 = numpy.round(numpy.log10(data['strain'].max()))
        scale = 1 / 10 ** pow10
        y_label = f'Strain x 10^{pow10}'
        data['strain'] *= scale

    fig = express.line(data, x='x', y='strain', color='polarization',
                       height=HEIGHT_LINE,
                       width=WIDTH_LINE)

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text=y_label)

    # Set minimal margin
    fig.update_layout(margin=PLOT_MARGIN)

    # Set legend location to below
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

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

    # Set legend location to below
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

    return fig


def cbc_spectrogram(data: xarray.DataArray, height=HEIGHT_IMSHOW, width=WIDTH_IMSHOW):
    """Plot a spectrogram of a CBC waveform for a given binary system.

    Args:
        data:
            xarray.DataArray, a spectrogram of a CBC waveform
    """
    fig = express.imshow(data, origin='lower',
                         height=height,
                         width=width)

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Frequency (Hz)')

    # Set minimal margin
    fig.update_layout(margin=PLOT_MARGIN)

    return fig


def snr_time_domain(data: pandas.DataFrame):
    """Plot a CBC waveform for a given binary system.

    Args:
        data:
            pandas.DataFrame, time domain strain data with columns: x, y
    """
    data = data.copy()

    # Scale y axis
    # max_snr = data['y'].abs().max()
    # if max_snr > 0:
    #     pow10 = numpy.round(numpy.log10(max_snr))
    #     scale = 1 / 10 ** pow10
    #     y_label = f'SNR x 10^{pow10}'
    #     data['y'] *= scale

    fig = express.line(data, x='x', y='y',
                       height=HEIGHT_LINE,
                       width=WIDTH_LINE)

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='SNR')

    # Set y-axis to be logscale
    fig.update_yaxes(type='log')

    # Set minimal margin
    fig.update_layout(margin=PLOT_MARGIN)

    return fig


def game_history(data: pandas.DataFrame):
    """Plot the game history of guesses and matches.

    Args:
        data:
            pandas.DataFrame, game history data with columns: m1, m2, s1z, s2z, approximant, polarization, guess_m1, guess_m2, guess_s1z, guess_s2z, match, mismatch, time
    """
    print('Plotting game history', data)
    fig = express.scatter(data, x='time', y='match',
                          height=2 * HEIGHT_LINE,
                          width=2 * HEIGHT_LINE,
                          opacity=0.6)

    # Update point size
    fig.update_traces(marker=dict(size=12))

    # Set axis labels
    fig.update_xaxes(title_text='Time (s)')
    fig.update_yaxes(title_text='Match')

    # Set axis limits
    fig.update_xaxes(range=[-0.1, max(60, data['time'].max() * 1.1)])
    fig.update_yaxes(range=[-0.1, 1.1])

    # Set minimal margin
    fig.update_layout(margin=PLOT_MARGIN)

    return fig
