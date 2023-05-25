import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, no_update, State, ctx, dash_table
from jupyter_dash import JupyterDash
from dash.exceptions import PreventUpdate
import base64
import dash_uploader as du
from utils import *
import pandas as pd


def create_layout(app, flists, config):
    
    return html.Div([
                    html.Br(),
                    # H&E image
                    dbc.Row([
                        html.H2('H&E segmentation'),
                        dbc.Col([
                            html.H4('Select sample and organism'),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="he_dropdown", 
                                        multi=False,
                                        options=flists, 
                                        placeholder = 'Select sample to analyze',
                                        className="app__dropdown",
                                        persistence = True, 
                                        persistence_type = 'memory',
                                        style = {
                                                    #'margin-top': '3%',
                                                   # 'margin-bottom': '3%',
                                                    'width': '100%',
                                                    'float' : 'center',
                                                    'min-height': '5px',
                                                    'height' : '5px',
                                                    'align' : 'left',
                                            },
                            )], width = 10),
                                dbc.Col([
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "Human", "value": "human"},
                                            {"label": "Mouse", "value": "mouse"}
                                        ],
                                        value="human",
                                        id="organism-radio",
                                        inline = True,
                                        labelStyle={'display': 'block'},
                                        style = {'align' : 'center',
                                                 'align-items' : 'center',
                                                 'vertical-align':'center',
                                                 'float' : 'center',
                                                 'margin-left': '10%',
                                                 'margin-right': '10%',},
                                    )], width = 2
                                )
                            ]),
                            html.Br(),
                            html.H4('Visium H&E image'),
                            # H&E Figure
                            html.Div(
                                dcc.Loading(id="ls-loading-2", 
                                            children=[
                                                        dcc.Graph(id='he_image', # 이 부분은 이미지가 호출되는 부분
                                                          figure=initial_img('assets/he_default.jpg'),
                                                          config = config['graph_config'],        
                                                          style={
                                                                    'margin-top': '0',
                                                                    'margin-bottom': '0',
                                                                    'height': '50vw',
                                                                    'paper_bgcolor' : 'black',
                                                                    'plot_bgcolor': 'black',
                                                                    'margin-left': 'auto',
                                                                    'margin-right': 'auto'
                                                                },
                                                        ),
                                            ],
                                        color= "#542C95",
                                        type="default"
                                    ),
                                id="hover_click",
                                n_clicks=0,
                                style = {
                                'align' : 'center',
                                },
                            ),
                                
                        ],width=8),

                        # Slider, button, DEGs
                        dbc.Col([
                            html.H4('Segment-anything (Prompt mode)'),
                            html.H6('Draw rectangles and press SAM button'),
                            # Button to Run SAM
                            html.Div(
                                html.P(id = 'box')
                            ),                                                
                            
                            html.Div(id = 'overlay_dropdown'),
                            html.Div(
                                [dbc.Button(
                                    color='primary',
                                    id='run_sam_prompt', # 동주님, 홍윤님 다름
                                    n_clicks=0, # 초기값에 불과함
                                    class_name="btn btn-lg btn-primary",
                                    children='Run SAM',
                                    style={
                                        'background-color': "#542C95"
                                    }
                                ),
                                dbc.Button(
                                    id='reset_prompt',
                                    outline=True, 
                                    color="secondary",
                                    children="Reset",
                                    n_clicks = 0,
                                    class_name = "btn btn-secondary"
                            )],  
                            className="d-grid gap-2"),   
                            html.Br(),
                            html.Hr(),
                            html.P('Mask opacity'),
                            dcc.Slider(
                                    id="alpha-state", 
                                    min=0, 
                                    max=1, 
                                    value=0.6, 
                                    step=0.1, 
                                    tooltip={
                                        'placement' : 'bottom',
                                        'always_visible':False
                                    },
                            ),
                            html.Hr(),
                            html.H4('Downstream analysis'),
                            dbc.Row([
                                dbc.Col([
                                    html.P('logFC cutoff'),
                                    dbc.Select(
                                        [0, 0.5, 1, 1.5, 2, 2.5, 3],
                                        id="lfc_cutoff", 
                                        value=1, 
                                )
                                ]),
                                dbc.Col([
                                    html.P('p-adj cutoff'),
                                    dbc.Select(
                                    [0.001, 0.01, 0.05, 0.1],
                                    id="pval_cutoff",
                                    value=0.05,
                                    )
                                ])
                            ]),
                            html.Br(),
                            html.Div([
                                html.P('Geneset for enrichment analysis'),
                                dbc.Checklist(
                                    id = 'geneset',
                                    options = [
                                        {'label' : 'GO_BP', 'value' : 'GO_Biological_Process_2018'}, 
                                        {'label' : 'GO_CC', 'value' : 'GO_Cellular_Component_2018'}, 
                                        {'label' : 'GO_MF', 'value' : 'GO_Molecular_Function_2018'},
                                        {'label' : 'MSigDB', 'value' :'MSigDB_Hallmark_2020'},
                                        {'label' : 'KEGG (Human)', 'value' : 'KEGG_2021_Human'}
                                    ],
                                    value = ['GO_Biological_Process_2018', 'GO_Cellular_Component_2018', 'GO_Molecular_Function_2018'],
                                   inline = True
                                )]
                            ),
                            html.Br(),
                            html.Div([
                                html.P('CellTypist Reference model'),
                                dcc.Dropdown(
                                        id="model_dropdown",
                                        multi=False,
                                        value = 'Immune_All_High.pkl',
                                        options = pd.read_csv('assets/celltypist_models_description.csv')['model'].tolist()
                                )
                            ]),
                            html.Br(),
                            # Run ST analysis
                            html.Div(
                                [dbc.Button(
                                    color='primary',
                                    id='run_deg', 
                                    n_clicks=0,
                                    class_name="btn btn-lg btn-primary",
                                    children='Run ST analysis',
                                    type="button",
                                    style={
                                        'background-color': "#542C95"
                                    }
                                ),
                                 dbc.Row([
                                     dbc.Col([
                                         dbc.Button(
                                            id='export-barcode',
                                            outline=True, 
                                            color="secondary",
                                            children="Export ROI barcode",
                                            n_clicks = 0,
                                            class_name = "btn btn-secondary"
                                        )
                                     ]),
                                     dbc.Col([
                                         dbc.Button(
                                            id='export-deg',
                                            outline=True, 
                                            color="secondary",
                                            children="Export DEG table",
                                            n_clicks = 0,
                                            class_name = "btn btn-secondary"
                                        )
                                     ])
                                 ]),
                                dcc.Download(id = 'download-barcode'),
                                dcc.Download(id = 'download-deg'),
                                html.Br(),
                                dbc.Spinner(html.Div(id = 'msg'))],  
                                className="d-grid gap-2"
                            )
                        ],width=4)
            ]),
            html.Br(),
            # DEG analysis and GO enrichment analysis
            html.Div([
                dbc.Accordion([
                     dbc.AccordionItem([
                        dbc.Row([
                                dcc.Loading(
                                    dcc.Graph(
                                            id="deg_volcano",
                                            figure= blank_fig(),
                                            style={
                                                    'margin-top': 'auto',
                                                    'margin-bottom': 'auto',
                                                    'margin-left': 'auto',
                                                    'margin-right': 'auto'
                                            },
                                    ),
                                    id = "loading-volcano",
                                    color= "#542C95",
                                    fullscreen= False,
                                    style={'background-color':'transparent'}
                                )
                        ]),
                        dbc.Row([
                                dcc.Loading(
                                    dcc.Graph(
                                        id="deg_box",
                                        figure = blank_fig(),
                                        style={
                                                'margin-top': 'auto',
                                                'margin-bottom': 'auto',
                                                'margin-left': 'auto',
                                                'margin-right': 'auto'
                                        },
                                    ),
                                    id = "loading-deg4",
                                    color= "#542C95",
                                    fullscreen= False,
                                    style={'background-color':'transparent'}
                                )
                        ])
                     ], title="Differentially Expressed Genes", id = "accord1"
                ),
                dbc.AccordionItem([         
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                dcc.Graph(
                                    id="deg_enrich",
                                    figure = blank_fig(),
                                    style={
                                            'margin-top': 'auto',
                                            'margin-bottom': 'auto',
                                            'margin-left': 'auto',
                                            'margin-right': 'auto'
                                    },
                                ),
                                id = "loading-deg3",
                                color= "#542C95",
                                fullscreen= False,
                                style={'background-color':'transparent'}
                            )
                        ]),

                    ]) # Row
                ], title ='Enrichment analysis' ), # Div
             dbc.AccordionItem([    
                 dbc.Row([
                    dcc.Loading(
                            dcc.Graph(
                                id="deg_celltype",
                                figure = blank_fig(),
                                style={
                                            'margin-top': 'auto',
                                            'margin-bottom': 'auto',
                                            'margin-left': 'auto',
                                            'margin-right': 'auto'
                                    },
                            ),
                            id = "loading-celltype",
                            color= "#542C95",
                            fullscreen= False,
                            style={'background-color':'transparent'}
                        )])], title = 'Cell type prediction')
            ], start_collapsed=True,  always_open=True)
        ]),

        ],
        className="app__container",
        style = {'margin-left' : '5%',
                          'margin-right' : '5%',
                        'margin-bottom': '5%'}
)
