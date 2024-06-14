"""GWave Game Application 0: Waveform Explorer
"""

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, Output, Input, html, State

import plot, waveforms

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

app = dash.Dash('CBC Waveform Explorer', external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets', assets_url_path='/assets/')
server = app.server

data = waveforms.get_cbc_waveform(50.0, 50.0, 0.0, 0.0)
data_spec = waveforms.get_spectrogram_data(data)
fig_td = plot.cbc_time_domain(data)
fig_fd = plot.cbc_freq_domain(data)
fig_spec = plot.cbc_spectrogram(data_spec)
waveforms.generate_audio_file(50.0, 50.0, 0.0, 0.0)

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
                        dbc.Label("Approximant"),
                        dcc.Dropdown(options=waveforms.APPROXIMANTS,
                                     value='IMRPhenomD', id='dropdown-approximant'),
                    ], md=2),
                    dbc.Col(children=[
                        dbc.Label("Polarization"),
                        dcc.Dropdown(options=['plus', 'cross', 'both'],
                                     value='plus', id='dropdown-polarization'),
                    ], md=2),
                    dbc.Col(children=[
                        html.Button('Make Audio', id='button-make-audio', n_clicks=0),
                    ], md=1),
                    dbc.Col(children=[
                        # Add toggle button for frequency shift
                        daq.ToggleSwitch(id='toggle-shift', value=False, label='Shift Frequency'),
                    ], md=1),
                    dbc.Col(children=[
                        dbc.Label("Play Audio"),
                        html.Div(id='audio-player-container')#, children=[
                            # html.Audio(id='audio-player', autoPlay=False, controls=True, style={'width': '100%'},
                            #            src='assets/50.0_50.0_0.0_0.0_IMRPhenomD_plus.wav', n_clicks=0, loop=False),
                        #]),
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

        ]),
        # Add placeholder for no-op javascript callback
        html.Div(id="placeholder", style={"display": "none"})

    ]),
])


@app.callback(
    Output('audio-player-container', 'children'),
    Input('button-make-audio', 'n_clicks'),
    State('slider-m1', 'value'),
    State('slider-m2', 'value'),
    State('slider-s1z', 'value'),
    State('slider-s2z', 'value'),
    State('dropdown-approximant', 'value'),
    State('dropdown-polarization', 'value'),
    State('toggle-shift', 'value'),
)
def generate_audio(n_clicks, m1, m2, s1z, s2z, approximant, polarization, shift):
    """Generate the audio waveform based on the input values."""
    print('Generating audio file')
    if n_clicks is None or n_clicks == 0:
        return []

    filepath = waveforms.generate_audio_file(m1, m2, s1z, s2z, approximant, polarization=polarization, return_filepath=True, shift_freq=shift)
    return [html.Audio(id='audio-player', autoPlay=False, controls=True, style={'width': '100%'},
                        src=filepath, n_clicks=0, loop=False)]
    # return [html.Label(f'Audio file generated: {n_clicks}')]


app.clientside_callback(
    """
    function(n) {
      var audio = document.querySelector('#audio-player');
      if (!audio){
        return -1;
      }
      audio.load();
      audio.play();
      return '';
   }
    """,
    Output('placeholder', 'children'),
    [Input('audio-player', 'n_clicks')],
    prevent_initial_call=True
)


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
    app.run_server(debug=False, host='0.0.0.0', port=9000)
