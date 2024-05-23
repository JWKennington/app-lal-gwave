"""This module provides a simplified interface to CBC waveforms using LALSuite.
"""

import lal
import lalsimulation
import numpy
import pandas

APPROXIMANTS = [
    'IMRPhenomD',
    'IMRPhenomPv2',
]


def get_cbc_waveform(m1: float, m2: float, s1z: float = 0.0, s2z: float = 0.0, approximant: str = 'IMRPhenomD') -> pandas.DataFrame:
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
    kwargs_td['deltaT'] = 1.0 / 256

    # Make frequency-domain kwargs
    kwargs_fd = kwargs.copy()
    kwargs_fd['f_max'] = 256.0
    kwargs_fd['deltaF'] = 1.0 / 64

    hplus_td, hcross_td = lalsimulation.SimInspiralTD(**kwargs_td)
    hplus_fd, hcross_fd = lalsimulation.SimInspiralFD(**kwargs_fd)

    # Make array of time coordinates
    ts = numpy.arange(0, len(hplus_td.data.data) * hplus_td.deltaT, hplus_td.deltaT)

    # Make array of frequency coordinates
    fs = numpy.arange(0, len(hplus_fd.data.data) * hplus_fd.deltaF, hplus_fd.deltaF)


    # Concat all data
    data = pandas.concat([
        pandas.DataFrame({'x': ts, 'strain': hplus_td.data.data}).assign(polarization='plus', domain='time', component='real'),
        pandas.DataFrame({'x': ts, 'strain': hcross_td.data.data}).assign(polarization='cross', domain='time', component='real'),
        pandas.DataFrame({'x': fs, 'strain': numpy.real(hplus_fd.data.data)}).assign(polarization='plus', domain='freq', component='real'),
        pandas.DataFrame({'x': fs, 'strain': numpy.imag(hplus_fd.data.data)}).assign(polarization='plus', domain='freq', component='imag'),
        pandas.DataFrame({'x': fs, 'strain': numpy.real(hcross_fd.data.data)}).assign(polarization='cross', domain='freq', component='real'),
        pandas.DataFrame({'x': fs, 'strain': numpy.imag(hcross_fd.data.data)}).assign(polarization='cross', domain='freq', component='imag'),
    ], axis=0)
    return data
