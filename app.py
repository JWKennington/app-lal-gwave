"""This module defined the Plotly Dash App for the web interface.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, Output, Input, html

import waveforms

# Configuration constants for app input boundaries
MASS_MIN = 1.0
MASS_MAX = 200.0
MASS_STEP = 10.0
SPIN_MIN = -0.99
SPIN_MAX = 0.99
SPIN_STEP = 0.2

app = dash.Dash('CBC Waveform Visualizer', external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

fig = waveforms.plot_cbc_waveform(50.0, 50.0, 0.0, 0.0)

app.layout = dbc.Container(children=[

    # Blank row for spacing
    dbc.Row(children=[dbc.Col(children=[html.Br()], md=12)]),

    # Row for all inputs
    dbc.Row(children=[

        # Row for title and approximant / Polarization dropdowns
        dbc.Row(children=[
            dbc.Col(children=[
                dbc.Label("CBC Waveform Visualizer", style={'font-size': '24px'}),
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
                dcc.Slider(min=MASS_MIN, max=MASS_MAX, step=MASS_STEP, value=50.0, id='slider-m2'),
            ], md=8),
            dbc.Col(children=[
                dbc.Label("S2z"),
                dcc.Slider(min=SPIN_MIN, max=SPIN_MAX, step=SPIN_STEP, value=0.0, id='slider-s2z'),
            ], md=4),
        ]),
    ]),

    # Row for the waveform plot
    dbc.Row(children=[
        dbc.Col(children=[
            dcc.Graph(id='graph-waveform', figure=fig),
        ], md=12)
    ]),

])


@app.callback(
    Output('graph-waveform', 'figure'),
    Input('slider-m1', 'value'),
    Input('slider-m2', 'value'),
    Input('slider-s1z', 'value'),
    Input('slider-s2z', 'value'),
    Input('dropdown-approximant', 'value'),
    Input('dropdown-polarization', 'value'),
)
def update_waveform(m1, m2, s1z, s2z, approximant, polarization):
    """Update the waveform plot based on the input values."""
    return waveforms.plot_cbc_waveform(m1, m2, s1z, s2z, approximant, polarization)


if __name__ == "__main__":
    app.run_server(debug=False)
