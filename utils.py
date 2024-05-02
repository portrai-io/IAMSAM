import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import cv2
import json
import mygene
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import celltypist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scanpy as sc
import gseapy
import os
from SAM import IAMSAM

encoded_logo = base64.b64encode(open("assets/Portrai_Primary_RGB.png", 'rb').read())
encoded_iamsamlogo = base64.b64encode(open("assets/logo.png", 'rb').read())

def load_json(configfile):
    with open(configfile) as f:
        config = json.load(f)
    return config 


def check_sample_folder(sample_folder):
    required_files = [
        'filtered_feature_bc_matrix.h5',
        'spatial/tissue_positions_list.csv',
        'spatial/scalefactors_json.json',
        'spatial/tissue_lowres_image.png',
        'spatial/tissue_hires_image.png'
    ]
    
    # Check if the sample folder exists
    if not os.path.isdir(sample_folder):
        return False
    
    # Check if all required files exist within the sample folder
    for file_path in required_files:
        if not os.path.isfile(os.path.join(sample_folder, file_path)):
            print(file_path)
            return False
    
    return True
    

# Using base64 encoding and decoding
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
      image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')



def plot_mask(blendimg, masks_int):
    
    fig = px.imshow(blendimg)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.update_traces(
        customdata = masks_int,
        hovertemplate="MaskNumber: %{customdata}<extra></extra>"
    )
    
    return fig




def Header(app):
    return html.Div([
                        get_header(app), 
                        get_menu(),
                    ], className="app__title",
                     style = {  # style for Header
                        'margin-left': '5%',
                        'margin-right': '5%',
                        'display': 'block'
                    })



def get_header(app):
    header = html.Div([
                    dbc.Row(
                        [dbc.Col(
                            html.A(
                                    href = "https://github.com/portrai-io/IAMSAM",
                                    target="_blank",
                                    rel="noopener",
                                    children = [
                                        html.Img(
                                            src='data:image/png;base64,{}'.format(encoded_iamsamlogo.decode()),
                                            id='title',
                                            style={
                                                    'float' : 'center',
                                                    'vertical-align' : 'middle',
                                                    'align' : 'center',
                                                    'height': '100px',
                                                  'display': 'block',
                                                  'margin-left': 'auto',
                                                  'margin-right': 'auto'
                                            }
                                        ),
                                    ],
                            )
                    )], 
                    style={
                        'height' : '100px'
                    }),
                    html.Div(
                        [
                            html.H5("About this app"),
                            html.P([
                                "IAMSAM (Image-based Analysis of Molecular signatures using the Segment-Anything Model) is a user-friendly web-based tool designed to analyze ST(Spatial Transcriptomics) data. To fully utilize the functionalities of IAMSAM, please refer to the resources below. If you have any questions, please contact with ",
                                html.A("contact@portrai.io", href = "mailto:contact@portrai.io")
                            ]),
                            html.Ul([
                                html.Li([html.A('Our paper', href = "https://www.biorxiv.org/content/10.1101/2023.05.25.542052v1")]),
                                html.Li([html.A('Github Repo', href = "https://github.com/portrai-io/IAMSAM")]),
                                html.Li([html.A('Tutorial video', href = "https://youtu.be/ri1OB4W210Q")])
                            ])] 
                ),
        html.Hr(),
        html.Div(
            [
                dbc.Offcanvas([
                    html.H5("Everything-mode"),
                    html.P("SAM performs image segmentation automatically without user input. The model generates segmentation masks, and the user can select masks with clicking it to choose regions of interest after the model generates the segmentation masks."),
                    html.Br(),
                    html.H5("Prompt-mode"),
                    html.P("SAM allows the user to input prompts by specifying rectangle boxes in the image that correspond to objects for segmentation. The model generates segmentation masks based on this input prompt.")],
                id="offcanvas",
                title="About Mode",
                is_open=False,
                )
            ]
)
        
        
    ])
    return header

def get_menu():
    return html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Nav([
                        dbc.NavLink(
                            "Everything-mode",
                            href="/",
                            active= "exact",
                            className="nav-item nav-link",
                        ),
                        dbc.NavLink(
                            "Prompt-mode",
                            href="/prompt",
                            active ="exact",
                            className="nav-item nav-link",
                        ),
                        ], pills=True,
                    className="nav nav-pills",
                    style = {
                        'font-size' : 'x-large',
                        'font-weight' : 'bold'
                    }
                ),width = 11
            ),
            dbc.Col(
                dbc.Button("About Mode", id="open-offcanvas", n_clicks=0, className="btn btn-secondary"),
                width = 1
            )
        ])
    ])
    

def footnote():
    footnote = html.Div([
            html.Div(
                html.P('Contact us for collaborations and research opportunities with Portrai, and stay tuned as we continue to improve!'),
            style={
                'text-align' : 'center',
                'align' : 'center'
                    }
            ),
       #     html.Div(id = 'reset_load', n_clicks = 0), # Hidden div to reset with 
            dbc.Row(
                dbc.Col(
                    html.A( 
                    href="https://portrai.io/", 
                    target="_blank",
                    rel="noopener",
                    children=[
                        html.Img(
                                src='data:image/png;base64,{}'.format(encoded_logo.decode()),
                                style={
                                    'height' : '30px',
                                    'vertical-align' : 'middle',
                                    'align' : 'center',
                                    'display': 'block',
                                    'margin-left': 'auto',
                                    'margin-right': 'auto'
                                }
                        ),
                    ],
                    )
                )
            )
    ])
            
    return footnote


def initial_img(path):
    initial_img = cv2.imread(path)
    initial_img = cv2.cvtColor(initial_img, cv2.COLOR_BGR2RGB)
    fig = px.imshow(initial_img)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    return fig

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig


def plot_volcano(In_df):
    fig = px.scatter(data_frame=In_df, x = "logfoldchanges", y = "-log10Padj",
                      text = "names", color = 'DE', color_discrete_map={
                          'ROI1' : '#542C95', 
                          'ROI2' : '#764a23',
                          'None' : 'lightgrey'
                      }, title = 'Volcano plot (ROI2 vs. ROI1)')
    fig.update_layout(legend_title_text = 'Differentially Expressed Gene')
    fig.update_traces(mode = "markers",
                     hovertemplate =  'Gene symbol : %{text} <br>' +
                                      'logFC : %{x} <br>' + 
                                      '-log10(P-adj) : %{y}') 
    fig.update_xaxes(title_text = 'logFC', showspikes=True, spikemode = "across", range=(-10,10))
    fig.update_yaxes(title_text = '-log10(P-adj)', showspikes=True, spikemode = "across")
    
    return fig

def plot_box(In_df, adata, top_n = 10):

    adata_roi = adata[np.isin(adata.obs['ROIs'], ['ROI1', 'ROI2']),:].copy()
    
    top_gene = In_df[In_df['DE'] == 'ROI1'].sort_values('logfoldchanges', ascending=False)
    top_gene = top_gene.head(n=top_n).names
    top_idx = [adata.var_names.tolist().index(x) if x in adata.var_names else None for x in top_gene]
    top_df = pd.DataFrame(adata.X.todense()[:,top_idx], columns=top_gene, index= adata_roi.obs_names)
    
    top_df = top_df.merge(adata_roi.obs['ROIs'], how = 'left', left_index=True, right_index=True)
    top_df['ROIs'] = top_df['ROIs'].astype('category')

    fig = px.box(data_frame = top_df.melt(id_vars = 'ROIs'), 
                    x = "variable",
                    y = 'value', color = 'ROIs',
                    color_discrete_map={
                          'ROI1' : '#542C95', 
                          'ROI2' : '#764a23'
                      }, 
                    category_orders={"mask_in" : ['ROI1', 'ROI2']},
                    title = 'Top{} high foldchange DEGs in ROI1 compare to ROI2'.format(top_n))
    fig.update_layout(legend_title_text = 'Region of Interest')
    fig.update_xaxes(title_text = 'Gene symbols')
    fig.update_yaxes(title_text = 'Normalized exp.')
    fig.show()
    
    return fig   


def do_enrichment_analysis_for_ROI1(In_df, gene_sets, organism, top_n = 10):
    degs_up = In_df[In_df['DE'] == 'ROI1'].names
    enr_up = gseapy.enrichr(gene_list = degs_up, 
                         gene_sets = gene_sets,
                         organism = organism,
                         outdir=None,
    )

    enr_up_filtered = enr_up.results[enr_up.results['Adjusted P-value'] < 0.05 ]
    enr_up_filtered['log10P'] = -np.log10(enr_up_filtered['Adjusted P-value'])
    enr_up_top = enr_up_filtered.sort_values('log10P', ascending=False).head(top_n)
    fig = px.bar(enr_up_top, 
             x = "log10P", y = "Term", color = 'Gene_set', title = 'Top enriched terms (adj.P < 0.05) in ROI1', 
             category_orders={'Term' : enr_up_top.Term.tolist()})
    fig.update_layout(legend_title_text = 'Gene set')
    fig.update_xaxes(title_text = '-log10(P-adj)')
    fig.update_yaxes(title_text = 'Terms')

    return fig


def do_enrichment_analysis_for_ROI2(In_df, gene_sets, organism, top_n = 10):
    degs_up = In_df[In_df['DE'] == 'ROI2'].names
    enr_up = gseapy.enrichr(gene_list = degs_up, 
                         gene_sets = gene_sets,
                         organism = organism,
                         outdir=None,
    )

    enr_up_filtered = enr_up.results[enr_up.results['Adjusted P-value'] < 0.05 ]
    enr_up_filtered['log10P'] = -np.log10(enr_up_filtered['Adjusted P-value'])
    enr_up_top = enr_up_filtered.sort_values('log10P', ascending=False).head(top_n)
    fig = px.bar(enr_up_top, 
             x = "log10P", y = "Term", color = 'Gene_set', title = 'Top enriched terms (adj.P < 0.05) in ROI2', 
             category_orders={'Term' : enr_up_top.Term.tolist()})
    fig.update_layout(legend_title_text = 'Gene set')
    fig.update_xaxes(title_text = '-log10(P-adj)')
    fig.update_yaxes(title_text = 'Terms')

    return fig
    


def plot_deconv_piechart(adata, roi):
    celltype_df = adata.obs.loc[adata.obs['ROIs'] == roi, adata.obs.columns.str.startswith('celltype')].copy()
    
    prop_df = pd.DataFrame(celltype_df.mean(axis=0))
    prop_df.index = prop_df.index.str.replace('celltype_', '')
    prop_df.columns = ['prop']
    
    fig = px.pie(data_frame=prop_df, names = prop_df.index, values='prop')

    fig.update_layout(legend_title_text = f'Cell type proportion of {roi}')
    fig.update_traces(textposition = 'inside', textinfo='percent+label',
                        hovertemplate =  '%{label} : %{percent} ') 
    return fig
    

    
# def do_celltypist(model, adata, sample_organism):
    
#     celltypist.models.download_models()
    
#     df = pd.read_csv('assets/celltypist_models_description.csv')
#     model_organism = df[df.model == model].organism.values[0]
    
#     if sample_organism == model_organism:
#         print("Skip gene conversion")
#     else:
#         adata = human_mouse_conversion(adata, model_organism)
    
#     predictions = celltypist.annotate(adata, 
#                                   model = model, 
#                                 mode = 'best match',
#     )
    
#     prob_mat = predictions.probability_matrix
#     prob_mat = prob_mat.apply(lambda x : x / x.sum(), axis = 1)
#     prob_mat = prob_mat.merge(adata.obs['ROIs'], left_index=True, right_index=True)
#     prob_mat_in = prob_mat[prob_mat['ROIs'] == 'ROI1']
#     cell_type_in = pd.DataFrame(prob_mat_in.sum(), columns = ['sum_prob']).reset_index()
    
#     fig = px.pie(data_frame=cell_type_in.sort_values('sum_prob', ascending=False).head(n = 6), 
#              names = "index", values='sum_prob')
#     fig.update_layout(legend_title_text = 'Top 6 cell type')
#     fig.update_traces(textposition = 'inside', textinfo='percent+label',
#                         hovertemplate =  '%{label} : %{percent} ') 
#     return fig
    
    
    
# def human_mouse_conversion(input_anndata, target_organism):
#     hom_table = pd.read_csv('assets/HOM_mouse_human_symbol_table.csv')
    
#     adata = input_anndata.copy()

#     #h_or_m = human_or_mouse(adata)
#     symbols = adata.var_names.tolist()

#     # Mouse -> Human
#     if target_organism == 'human':
#         print("Gene symbol conversion : Mouse -> Human")
#         hom_mouse_symbol = hom_table['Mouse_symbol'].tolist()
#         for i in range(len(symbols)):
#             gene = symbols[i]
#             if gene in hom_mouse_symbol:
#                 gene_h = hom_table.iloc[hom_mouse_symbol.index(gene), 2]
#                 symbols[i] = gene_h

#     # Human -> Mouse
#     elif target_organism == 'mouse' :
#         print("Gene symbol conversion : Human -> Mouse")
#         hom_human_symbol = hom_table['Human_symbol'].tolist()
#         for i in range(len(symbols)):
#             gene = symbols[i]
#             if gene in hom_human_symbol:
#                 gene_m = hom_table.iloc[hom_human_symbol.index(gene), 1]
#                 symbols[i] = gene_m

#     else:
#         print("Celltypist can only analyze human or mouse.")


#     adata.var_names = symbols
#     adata.var_names = adata.var_names.astype('str')
#     return adata