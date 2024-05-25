"""This module defined the Plotly Dash App for the web interface.

Game: Visual Parameter Estimation!
"""
import datetime

import dash
import dash_bootstrap_components as dbc
import numpy.random
import pandas
from dash import dcc, Output, Input, html, State

import plot
import waveforms

# Configuration constants for app input boundaries
MASS_MIN = 1.0
MASS_MAX = 200.0
MASS_STEP = 10.0
SPIN_MIN = -0.99
SPIN_MAX = 0.99
SPIN_STEP = 0.2

BACKGROUND_IMAGE = {
    'background-image': 'url("/assets/Ripple.png")',
    'background-position': 'center',
    'background-size': 'cover',
}

BACKGROUND_WHITE = {
    'background-color': 'white',
}

GAME_HISTORY = pandas.DataFrame(columns=['m1', 'm2', 's1z', 's2z', 'approximant', 'polarization',
                                         'guess_m1', 'guess_m2', 'guess_s1z', 'guess_s2z', 'mismatch', 'time'])
GAME_EVENT_PARAMS = None
GAME_EVENT_START = None

app = dash.Dash('CBC Waveform Explorer', external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets', assets_url_path='/assets/')
server = app.server

data = waveforms.get_cbc_waveform(50.0, 50.0, 0.0, 0.0)
data_spec = waveforms.get_spectrogram_data(data)
fig_td = plot.cbc_time_domain(data)
fig_fd = plot.cbc_freq_domain(data)
fig_spec = plot.cbc_spectrogram(data_spec)

app.layout = dbc.Container(fluid=False, children=[

    html.Div(style=BACKGROUND_WHITE, children=[
        dbc.Row(children=[

            # Blank row for spacing
            dbc.Row(children=[dbc.Col(children=[html.Br()], md=12)]),

            # Row for all inputs
            dbc.Row(style=BACKGROUND_WHITE, children=[

                # Row for title and approximant / Polarization dropdowns
                dbc.Row(children=[
                    dbc.Col(children=[
                        dbc.Label("CBC Waveform Explorer", style={'font-size': '24px'}),
                    ], md=4),
                    dbc.Col(children=[
                        html.Button('New Game', id='button-new-game', n_clicks=0),
                    ], md=2),
                    dbc.Col(children=[
                        html.Button('New Event', id='button-new-event', n_clicks=0),
                    ], md=2),
                    dbc.Col(children=[
                        html.Button('Lock Guess', id='button-lock-guess', n_clicks=0),
                    ], md=2),
                    dbc.Col(children=[
                        dbc.Label("Polarization"),
                        dcc.Dropdown(options=['plus', 'cross', 'both'],
                                     value='plus', id='dropdown-polarization'),
                    ], md=2),

                    # Add button to generate random parameters for fake data
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

        ]),
        # Dummy div for no-op callbacks
        html.Div(id="placeholder", style={"display": "none"})
    ]),
])


@app.callback(
    Output('placeholder', 'children'),
    Input('button-new-game', 'n_clicks'),
)
def new_game(n_clicks):
    """Reset the game history."""
    print('New Game')
    # Yeah, I know 'global' is bad, but it's either that or embed the game state in useless widgets, so...
    global GAME_EVENT_START, GAME_EVENT_PARAMS
    GAME_HISTORY.drop(GAME_HISTORY.index, inplace=True)
    GAME_EVENT_PARAMS = None
    GAME_EVENT_START = None


@app.callback(
    Output('placeholder', 'children'),
    Input('button-new-event', 'n_clicks'),
)
def new_event(n_clicks):
    """Reset the game history."""
    print('New Event')
    global GAME_EVENT_START, GAME_EVENT_PARAMS
    GAME_EVENT_PARAMS = {
        'm1': numpy.random.rand() * (MASS_MAX - MASS_MIN) + MASS_MIN,
        'm2': numpy.random.rand() * (MASS_MAX - MASS_MIN) + MASS_MIN,
        's1z': numpy.random.rand() * (SPIN_MAX - SPIN_MIN) + SPIN_MIN,
        's2z': numpy.random.rand() * (SPIN_MAX - SPIN_MIN) + SPIN_MIN,
        'approximant': 'IMRPhenomPv2',
        'polarization': 'plus',
    }
    GAME_EVENT_START = datetime.datetime.now()
    GAME_HISTORY.drop(GAME_HISTORY.index, inplace=True)


@app.callback(
    Output('placeholder', 'children'),
    Input('button-lock-guess', 'n_clicks'),
    State('slider-m1', 'value'),
    State('slider-m2', 'value'),
    State('slider-s1z', 'value'),
    State('slider-s2z', 'value'),
)
def lock_guess(n_clicks, m1, m2, s1z, s2z):
    """Reset the game history."""
    print('Lock Guess')
    global GAME_EVENT_START, GAME_EVENT_PARAMS
    if GAME_EVENT_PARAMS is None or GAME_EVENT_START is None:
        return

    guess_duration = (datetime.datetime.now() - GAME_EVENT_START).total_seconds()
    guess_params = {
        'm1': m1,
        'm2': m2,
        's1z': s1z,
        's2z': s2z,
    }
    GAME_EVENT_START = None
    GAME_HISTORY.drop(GAME_HISTORY.index, inplace=True)


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
