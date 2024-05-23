"""This module defined the Plotly Dash App for the web interface.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, Output, Input, html

import waveforms, plot

# Configuration constants for app input boundaries
MASS_MIN = 1.0
MASS_MAX = 200.0
MASS_STEP = 10.0
SPIN_MIN = -0.99
SPIN_MAX = 0.99
SPIN_STEP = 0.2

BACKGROUND_STYLE = {
    'background-image': 'url(“/assets/Ripple.png”)',
    'background-repeat': 'no-repeat',
    'background-position': 'right top',
    'background-size': '150px 100px',
}

app = dash.Dash('CBC Waveform Explorer', external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

data = waveforms.get_cbc_waveform(50.0, 50.0, 0.0, 0.0)
data_spec = waveforms.get_spectrogram_data(data)
fig_td = plot.cbc_time_domain(data)
fig_fd = plot.cbc_freq_domain(data)
fig_spec = plot.cbc_spectrogram(data_spec)

app.layout = dbc.Container(style=BACKGROUND_STYLE, children=[

    # Blank row for spacing
    dbc.Row(children=[dbc.Col(children=[html.Br()], md=12)]),

    # Row for all inputs
    dbc.Row(children=[

        # Row for title and approximant / Polarization dropdowns
        dbc.Row(children=[
            dbc.Col(children=[
                dbc.Label("CBC Waveform Explorer", style={'font-size': '24px'}),
            ], md=8),
            dbc.Col(children=[
                dbc.Label("Approximant"),
                dcc.Dropdown(options=waveforms.APPROXIMANTS,
                             value='IMRPhenomD', id='dropdown-approximant'),
            ], md=2),
            dbc.Col(children=[
                dbc.Label("Polarization"),
                dcc.Dropdown(options=['plus', 'cross', 'both'],
                             value='plus', id='dropdown-polarization'),
            ], md=2),
        ]),

        # Row for the first component inputs
        dbc.Row(children=[
            dbc.Col(children=[
                dbc.Label("M1 (M_sol)"),
                dcc.Slider(min=MASS_MIN, max=MASS_MAX, step=MASS_STEP, value=50.0, id='slider-m1', updatemode='drag'),
            ], md=8),
            dbc.Col(children=[
                dbc.Label("S1z"),
                dcc.Slider(min=SPIN_MIN, max=SPIN_MAX, step=SPIN_STEP, value=0.0, id='slider-s1z', updatemode='drag'),
            ], md=4),
        ]),

        # Row for the second component inputs
        dbc.Row(children=[
            dbc.Col(children=[
                dbc.Label("M2 (M_sol)"),
                dcc.Slider(min=MASS_MIN, max=MASS_MAX, step=MASS_STEP, value=50.0, id='slider-m2', updatemode='drag'),
            ], md=8),
            dbc.Col(children=[
                dbc.Label("S2z"),
                dcc.Slider(min=SPIN_MIN, max=SPIN_MAX, step=SPIN_STEP, value=0.0, id='slider-s2z', updatemode='drag'),
            ], md=4),
        ]),
    ]),

    # Row for the waveform plot
    dbc.Row(children=[
        dbc.Col(children=[
            dbc.Row(children=[
                dbc.Col(children=[
                    dcc.Graph(id='graph-waveform-td', figure=fig_td),
                ], md=12),
            ]),
            dbc.Row(children=[
                dbc.Col(children=[
                    dcc.Graph(id='graph-waveform-fd', figure=fig_fd),
                ], md=12),
            ]),
        ], md=7),
        dbc.Col(children=[
            dcc.Graph(id='graph-waveform-sp', figure=fig_spec),
        ], md=5),
    ]),

])


@app.callback(
    Output('graph-waveform-td', 'figure'),
    Output('graph-waveform-fd', 'figure'),
    Output('graph-waveform-sp', 'figure'),
    Input('slider-m1', 'value'),
    Input('slider-m2', 'value'),
    Input('slider-s1z', 'value'),
    Input('slider-s2z', 'value'),
    Input('dropdown-approximant', 'value'),
    Input('dropdown-polarization', 'value'),
)
def update_waveform(m1, m2, s1z, s2z, approximant, polarization):
    """Update the waveform plot based on the input values."""
    data = waveforms.get_cbc_waveform(m1, m2, s1z, s2z, approximant)
    data_spec = waveforms.get_spectrogram_data(data)
    fig_td = plot.cbc_time_domain(data, polarization)
    fig_fd = plot.cbc_freq_domain(data, polarization)
    fig_spec = plot.cbc_spectrogram(data_spec)
    return fig_td, fig_fd, fig_spec


if __name__ == "__main__":
    app.run_server(debug=False)
