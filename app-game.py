"""This module defined the Plotly Dash App for the web interface.

Game: Visual Parameter Estimation!
"""

# TODO update buttons to share callback according to: https://dash.plotly.com/duplicate-callback-outputs
# Maybe, maybe just set output values of sliders back to default on new event and that may trigger the second callback auto updating SNR plot

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

DIFFICULTY_NOISE_RATIO = {
    '1: Test': 0.0000001,
    '2: Easy': 0.01,
    '3: Medium': 0.1,
    '4: Hard': 0.5,
    '5: Expert': 1.0,
}

GAME_HISTORY = pandas.DataFrame(columns=['m1', 'm2', 's1z', 's2z', 'approximant', 'polarization',
                                         'guess_m1', 'guess_m2', 'guess_s1z', 'guess_s2z', 'match', 'mismatch', 'time'])
GAME_EVENT_PARAMS = None
GAME_EVENT_START = None
GAME_EVENT_DATA = waveforms.get_fake_data(duration=60, noise_scale=1.0, signal_scale=0.0, m1=50.0, m2=50.0, s1z=0.0, s2z=0.0)

app = dash.Dash('CBC Waveform Game', external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets', assets_url_path='/assets/')
server = app.server

fig_data = plot.cbc_time_domain(GAME_EVENT_DATA, scale=True)
fig_snr = plot.snr_time_domain(waveforms.get_snr_data(m1=50.0, m2=50.0, s1z=0.0, s2z=0.0, data=GAME_EVENT_DATA))

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
                        dbc.Label("CBC Waveform Game", style={'font-size': '24px'}),
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
                        dbc.Label("Difficulty"),
                        dcc.Dropdown(options=list(sorted(DIFFICULTY_NOISE_RATIO.keys())),
                                     value='2: Easy', id='dropdown-difficulty'),
                    ], md=2),

                    # Add button to generate random parameters for fake data
                ]),

                # Row for the first component inputs
                dbc.Row(children=[
                    dbc.Col(children=[
                        dbc.Label("M1 (M_sol)"),
                        dcc.Slider(min=MASS_MIN, max=MASS_MAX, step=MASS_STEP, value=100.0, id='slider-m1'),
                    ], md=8),
                    dbc.Col(children=[
                        dbc.Label("S1z"),
                        dcc.Slider(min=SPIN_MIN, max=SPIN_MAX, step=SPIN_STEP, value=0.0, id='slider-s1z'),
                    ], md=4),
                ]),

                # Row for the second component inputs
                dbc.Row(children=[
                    dbc.Col(children=[
                        dbc.Label("M2 (M_sol)"),
                        dcc.Slider(min=MASS_MIN, max=MASS_MAX, step=MASS_STEP, value=100.0, id='slider-m2'),
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
                    dbc.Row(children=[
                        dbc.Col(children=[
                            dcc.Graph(id='graph-data-raw', figure=fig_data),
                        ], md=12),
                    ]),
                    dbc.Row(children=[
                        dbc.Col(children=[
                            dcc.Graph(id='graph-data-snr', figure=fig_snr),
                        ], md=12),
                    ]),
                ], md=7),
            ]),

        ]),
        # Dummy div for no-op callbacks
        html.Div(id="placeholder", style={"display": "none"}),
        html.Div(id="placeholder2", style={"display": "none"}),
        html.Div(id="placeholder3", style={"display": "none"}),
        html.Div(id="placeholder4", style={"display": "none"}),
    ]),
])


@app.callback(
    Output('placeholder', 'children'),
    Input('button-new-game', 'n_clicks'),
)
def new_game(n_clicks):
    """Reset the game history."""
    if n_clicks is None or n_clicks == 0:
        return
    print('New Game')
    # Yeah, I know 'global' is bad, but it's either that or embed the game state in useless widgets, so...
    global GAME_EVENT_START, GAME_EVENT_PARAMS
    GAME_HISTORY.drop(GAME_HISTORY.index, inplace=True)
    GAME_EVENT_PARAMS = None
    GAME_EVENT_START = None


@app.callback(
    Output('placeholder2', 'children'),
    Output('graph-data-raw', 'figure'),
    Input('button-new-event', 'n_clicks'),
    State('dropdown-difficulty', 'value'),
)
def new_event(n_clicks, difficulty):
    """Reset the game history."""
    if n_clicks is None or n_clicks == 0:
        return None, fig_data
    print('New Event')
    global GAME_EVENT_START, GAME_EVENT_PARAMS, GAME_EVENT_DATA
    GAME_EVENT_PARAMS = {
        'm1': numpy.random.rand() * (MASS_MAX - MASS_MIN) + MASS_MIN,
        'm2': numpy.random.rand() * (MASS_MAX - MASS_MIN) + MASS_MIN,
        's1z': numpy.random.rand() * (SPIN_MAX - SPIN_MIN) + SPIN_MIN,
        's2z': numpy.random.rand() * (SPIN_MAX - SPIN_MIN) + SPIN_MIN,
        'approximant': 'IMRPhenomPv2',
        'polarization': 'plus',
    }

    # Round m1, m2, s1z, s2z to nearest slider values
    GAME_EVENT_PARAMS['m1'] = round(GAME_EVENT_PARAMS['m1'] / MASS_STEP) * MASS_STEP
    GAME_EVENT_PARAMS['m2'] = round(GAME_EVENT_PARAMS['m2'] / MASS_STEP) * MASS_STEP
    GAME_EVENT_PARAMS['s1z'] = round(GAME_EVENT_PARAMS['s1z'] / SPIN_STEP) * SPIN_STEP
    GAME_EVENT_PARAMS['s2z'] = round(GAME_EVENT_PARAMS['s2z'] / SPIN_STEP) * SPIN_STEP

    print(GAME_EVENT_PARAMS)

    # Update game params
    GAME_EVENT_START = datetime.datetime.now()

    # Update the waveform plot
    data = waveforms.get_fake_data(duration=60, noise_scale=DIFFICULTY_NOISE_RATIO[difficulty], **GAME_EVENT_PARAMS)
    GAME_EVENT_DATA = data

    # Make the plot
    fig = plot.cbc_time_domain(data)

    return None, fig


@app.callback(
    Output('placeholder3', 'children'),
    Input('button-lock-guess', 'n_clicks'),
    State('slider-m1', 'value'),
    State('slider-m2', 'value'),
    State('slider-s1z', 'value'),
    State('slider-s2z', 'value'),
)
def lock_guess(n_clicks, m1, m2, s1z, s2z):
    """Reset the game history."""
    if n_clicks is None or n_clicks == 0:
        return

    print('Lock Guess')
    global GAME_EVENT_START, GAME_EVENT_PARAMS, GAME_HISTORY
    if GAME_EVENT_PARAMS is None or GAME_EVENT_START is None:
        return

    guess_duration = (datetime.datetime.now() - GAME_EVENT_START).total_seconds()
    guess_params = {
        'm1': m1,
        'm2': m2,
        's1z': s1z,
        's2z': s2z,
    }
    event_params = GAME_EVENT_PARAMS.copy()
    event_params.pop('polarization')
    guess_match = waveforms._waveform_match(w1=waveforms._waveform_fd(**event_params)[0 if GAME_EVENT_PARAMS['polarization'] == 'plus' else 1],
                                            w2=waveforms._waveform_fd(**guess_params)[0 if GAME_EVENT_PARAMS['polarization'] == 'plus' else 1])

    # Record the guess
    GAME_HISTORY = pandas.concat([GAME_HISTORY, pandas.DataFrame({
        'm1': event_params['m1'],
        'm2': event_params['m2'],
        's1z': event_params['s1z'],
        's2z': event_params['s2z'],
        'approximant': event_params['approximant'],
        'polarization': GAME_EVENT_PARAMS['polarization'],
        'guess_m1': guess_params['m1'],
        'guess_m2': guess_params['m2'],
        'guess_s1z': guess_params['s1z'],
        'guess_s2z': guess_params['s2z'],
        'match': guess_match,
        'mismatch': 1 - guess_match,
        'time': guess_duration,
    }, index=[0])], axis=0)

    print(GAME_HISTORY)

    # Reset the event state
    GAME_EVENT_START = None
    GAME_EVENT_PARAMS = None

    # TODO update the game performance plots!


@app.callback(
    Output('graph-data-snr', 'figure'),
    Input('slider-m1', 'value'),
    Input('slider-m2', 'value'),
    Input('slider-s1z', 'value'),
    Input('slider-s2z', 'value'),
)
def update_waveform(m1, m2, s1z, s2z):
    """Update the waveform plot based on the input values."""
    print('Updating waveform plot')

    data = waveforms.get_snr_data(m1, m2, s1z, s2z, data=GAME_EVENT_DATA)
    fig_snr = plot.snr_time_domain(data)

    return fig_snr


if __name__ == "__main__":
    app.run_server(debug=False)
