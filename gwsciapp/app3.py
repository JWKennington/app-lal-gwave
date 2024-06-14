"""GWave Game Application 3: Parameter Estimation Game with spectrogram
"""

# TODO update buttons to share callback according to: https://dash.plotly.com/duplicate-callback-outputs
# Maybe, maybe just set output values of sliders back to default on new event and that may trigger the second callback auto updating SNR plot

import datetime

import dash
import dash_bootstrap_components as dbc
import numpy.random
import pandas
from dash import dcc, Output, Input, html, State, ctx

from gwsciapp import plot, waveforms

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

SPEC_HEIGHT = 250
SPEC_WIDTH = 700

DIFFICULTY_NOISE_RATIO = {
    '1: Test': 0.0000001,
    '2: Easy': 0.01,
    '3: Medium': 0.1,
    '4: Hard': 0.5,
    '5: Expert': 1.0,
}

GAME_DURATION = 20
GAME_HISTORY = pandas.DataFrame(columns=['m1', 'm2', 's1z', 's2z', 'approximant', 'polarization',
                                         'guess_m1', 'guess_m2', 'guess_s1z', 'guess_s2z', 'match', 'mismatch', 'time'])
GAME_EVENT_PARAMS = None
GAME_EVENT_START = None
GAME_EVENT_DATA = waveforms.get_fake_data(duration=60, noise_scale=1.0, signal_scale=0.0, m1=50.0, m2=50.0, s1z=0.0, s2z=0.0)

app = dash.Dash('CBC Detection Game: 4 (SNR)', external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets', assets_url_path='/assets/')
server = app.server

fig_data = plot.cbc_spectrogram(waveforms.get_spectrogram_data(GAME_EVENT_DATA), height=SPEC_HEIGHT, width=SPEC_WIDTH)
fig_template = plot.cbc_spectrogram(waveforms.get_spectrogram_data(waveforms.get_fake_data(duration=60, noise_scale=1.0, signal_scale=0.0, m1=50.0, m2=50.0, s1z=0.0, s2z=0.0)), height=SPEC_HEIGHT, width=SPEC_WIDTH)
fig_history = plot.game_history(GAME_HISTORY)

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
                        dbc.Label("CBC PE Level 3: Spectrogram", style={'font-size': '24px'}),
                    ], md=4),
                    dbc.Col(children=[
                        dbc.Button('New Game', id='button-new-game', n_clicks=0),
                    ], md=2),
                    dbc.Col(children=[
                        dbc.Button('New Event', id='button-new-event', n_clicks=0),
                    ], md=2),
                    dbc.Col(children=[
                        dbc.Button('Lock Guess', id='button-lock-guess', n_clicks=0),
                    ], md=2),
                    dbc.Col(children=[
                        dbc.Label("Difficulty"),
                        dcc.Dropdown(options=list(sorted(DIFFICULTY_NOISE_RATIO.keys())),
                                     value='3: Medium', id='dropdown-difficulty'),
                    ], md=2),

                    # Add button to generate random parameters for fake data
                ]),

                # Row for the first component inputs
                dbc.Row(children=[
                    dbc.Col(children=[
                        dbc.Label("M1 (M_sol)"),
                        dcc.Slider(min=MASS_MIN, max=MASS_MAX, step=MASS_STEP, value=100.0, id='slider-m1', updatemode='drag'),
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
                        dcc.Slider(min=MASS_MIN, max=MASS_MAX, step=MASS_STEP, value=100.0, id='slider-m2', updatemode='drag'),
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
                            dcc.Graph(id='graph-data-raw', figure=fig_data),
                        ], md=12),
                    ]),
                    dbc.Row(children=[
                        dbc.Col(children=[
                            dcc.Graph(id='graph-data-template', figure=fig_template),
                        ], md=12),
                    ]),
                ], md=7),
                dbc.Col(children=[
                    dcc.Graph(id='graph-data-history', figure=fig_history),
                ], md=5),
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
    # Combined outputs for all buttons
    Output('graph-data-raw', 'figure'),
    Output('graph-data-history', 'figure'),
    # All buttons serve as inputs (dispatched based on ctx.triggered_id)
    Input('button-new-game', 'n_clicks'),
    Input('button-new-event', 'n_clicks'),
    Input('button-lock-guess', 'n_clicks'),
    # Include all necessary states
    State('dropdown-difficulty', 'value'),
    State('slider-m1', 'value'),
    State('slider-m2', 'value'),
    State('slider-s1z', 'value'),
    State('slider-s2z', 'value'),
    State('graph-data-raw', 'figure'),
    State('graph-data-history', 'figure'),
)
def button_callback(n_clicks_new_game, n_clicks_new_event, n_clicks_lock_guess, difficulty, m1, m2, s1z, s2z, fig_data, fig_history):
    triggered_id = ctx.triggered_id

    if triggered_id == 'button-new-game':
        return sub_callback_new_game(n_clicks_new_game, fig_data, fig_history)

    elif triggered_id == 'button-new-event':
        return sub_callback_new_event(n_clicks_new_event, difficulty, fig_data, fig_history)

    elif triggered_id == 'button-lock-guess':
        return sub_callback_lock_guess(n_clicks_lock_guess, m1, m2, s1z, s2z, fig_data, fig_history)

    else:
        print('Unknown button', triggered_id)
        return fig_data, fig_history


def sub_callback_new_game(n_clicks, fig_data, fig_history):
    print('New Game')
    # Only reset the game history if the button was clicked, not on load
    if n_clicks is not None and n_clicks > 0:
        global GAME_HISTORY, GAME_EVENT_PARAMS, GAME_EVENT_START
        GAME_HISTORY.drop(GAME_HISTORY.index, inplace=True)
        GAME_EVENT_PARAMS = None
        GAME_EVENT_START = None

        fig_history = plot.game_history(GAME_HISTORY)

    return fig_data, fig_history


def sub_callback_new_event(n_clicks, difficulty, fig_data, fig_history):
    print('New Event')
    if n_clicks is not None and n_clicks > 0:
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
        data = waveforms.get_fake_data(duration=GAME_DURATION, noise_scale=DIFFICULTY_NOISE_RATIO[difficulty], **GAME_EVENT_PARAMS)
        GAME_EVENT_DATA = data

        # Make the plot
        fig_data = plot.cbc_spectrogram(waveforms.get_spectrogram_data(data), height=SPEC_HEIGHT, width=SPEC_WIDTH)

    return fig_data, fig_history


def sub_callback_lock_guess(n_clicks, m1, m2, s1z, s2z, fig_data, fig_history):
    print('Lock Guess')
    if n_clicks is not None and n_clicks > 0:
        global GAME_EVENT_START, GAME_EVENT_PARAMS, GAME_HISTORY
        if GAME_EVENT_PARAMS is None or GAME_EVENT_START is None:
            return fig_data, fig_history

        guess_duration = (datetime.datetime.now() - GAME_EVENT_START).total_seconds()
        guess_params = {
            'm1': m1,
            'm2': m2,
            's1z': s1z,
            's2z': s2z,
        }
        event_params = GAME_EVENT_PARAMS.copy()
        event_params.pop('polarization')
        guess_match = waveforms._match_at_coords(m1=event_params['m1'], m2=event_params['m2'], s1z=event_params['s1z'], s2z=event_params['s2z'],
                                                 guess_m1=guess_params['m1'], guess_m2=guess_params['m2'], guess_s1z=guess_params['s1z'], guess_s2z=guess_params['s2z'])

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

        # Reset the event state
        GAME_EVENT_START = None
        GAME_EVENT_PARAMS = None

        # Update the history plot
        fig_history = plot.game_history(GAME_HISTORY)

    return fig_data, fig_history


@app.callback(
    Output('graph-data-template', 'figure'),
    Input('slider-m1', 'value'),
    Input('slider-m2', 'value'),
    Input('slider-s1z', 'value'),
    Input('slider-s2z', 'value'),
    State('dropdown-difficulty', 'value'),
    State('button-new-event', 'n_clicks'),
)
def update_template(m1, m2, s1z, s2z, difficulty, n_clicks):
    """Update the waveform plot based on the input values."""
    if n_clicks is None or n_clicks == 0:
        return fig_template

    data = waveforms.get_fake_data(duration=GAME_DURATION, noise_scale=DIFFICULTY_NOISE_RATIO[difficulty],
                                   m1=m1, m2=m2, s1z=s1z, s2z=s2z)
    new_fig_template = plot.cbc_spectrogram(waveforms.get_spectrogram_data(data), height=SPEC_HEIGHT, width=SPEC_WIDTH)

    return new_fig_template


if __name__ == "__main__":
    app.run_server(debug=False)
