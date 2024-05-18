"""This module provides a simplified interface to CBC waveforms using LALSuite.
"""

import lal
import lalsimulation
import numpy
import pandas
from plotly import express

APPROXIMANTS = [
    'IMRPhenomD'
]


def get_cbc_waveform(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0) -> pandas.DataFrame:
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
        'approximant': lalsimulation.GetApproximantFromString('IMRPhenomD'),

        # Time domain parameters
        'deltaT': 1.0 / 256,

        # Frequency domain parameters
        'f_min': 15.0,
        # 'f_max': 2048.0,
        'f_ref': 0.0,
        # 'deltaF': 1.0 / 64,

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

    print(len(kwargs.keys()))
    hplus, hcross = lalsimulation.SimInspiralTD(**kwargs)

    # Make array of time coordinates
    ts = numpy.arange(0, len(hplus.data.data) * hplus.deltaT, hplus.deltaT)
    data = pandas.concat([
        pandas.DataFrame({'time': ts, 'strain': hplus.data.data}).assign(polarization='plus'),
        pandas.DataFrame({'time': ts, 'strain': hcross.data.data}).assign(polarization='cross'),
    ], axis=0)
    return data


def plot_cbc_waveform(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0):
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
    data = get_cbc_waveform(m1, m2, s1z, s2z)

    fig = express.line(data, x='time', y='strain', color='polarization', title='CBC Waveform')

    return fig
