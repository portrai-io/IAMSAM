#!/usr/bin/env python
"""
This is the core app code of IAMSAM
dash app 
"""


import os
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, no_update, State, ctx
from dash.exceptions import PreventUpdate
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors

from pages import everything, prompt
from SAM import IAMSAM
from utils import *
from log import log, get_log_messages


# Load Configuration
config = load_json('config/config.json')


# Load h5ad data from 'data' folder
data_files = os.listdir('data/')
data_files.sort()

flists = []
for file in data_files:
    if file.endswith('.h5ad'):
        flists.append(os.path.join('data/', file))
    
# App Layout
app = Dash(__name__, 
           meta_tags = [{'name': 'viewport', 'content': 'width=device-width'}],
           external_stylesheets = [dbc.themes.PULSE],
          suppress_callback_exceptions=True )
app.title = 'IAMSAM'
app._favicon = ("favicon.ico") 


app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), 
     Header(app),
     html.Div(id="page-content"),
     footnote()
    ]
)



### For layout ###

@app.callback(
    Output("log-textarea", "value"),
    Input("log-update-interval", "n_intervals")
)
def update_log_messages(n_intervals):
    return get_log_messages()


@app.callback(Output("page-content", "children"), 
              [Input("url", "pathname")])
def display_page(pathname):
    
    mode = "Everything-mode" if pathname != "/prompt" else "Prompt-mode"
    log(f"Mode changed to: {mode}")
    
    if pathname == "/prompt":
        return prompt.create_layout(app, flists, config)
    else:
        return everything.create_layout(app, flists, config)


@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('mask_list', 'value', allow_duplicate=True),
    Input('he_dropdown', 'value'), 
    State("reset", "n_clicks"),
    prevent_initial_call=True
)
def load_he_image(tissue_dir, reset):
    if tissue_dir is None:
        raise PreventUpdate
    reset = reset+1
    ps.adata = None
    ps.load_data(tissue_dir)
    fig = px.imshow(ps.tsimg_rgb)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    
    ps.boxes = []    
    return fig, ['']


@app.callback(
    Output('roi-1', 'options', allow_duplicate=True),
    Output('roi-2', 'options', allow_duplicate=True),
    Input('mask_list', 'options'),
    Input('roi-1', 'value'),
    Input('roi-2', 'value'),
    prevent_initial_call=True
)
def update_dropdown_options(master_values, selected1, selected2):
    all_options = [{'label': str(i), 'value': i} for i in range(1, len(ps.masks)+1)]

    if not master_values:
        return [], []

    options1 = [option for option in all_options if option['value'] not in (selected2 or [])]
    options2 = [option for option in all_options if option['value'] not in (selected1 or [])]
    
    return options1, options2

@app.callback(
    Output('mask_list', 'options', allow_duplicate=True),
    Output('mask_list', 'value', allow_duplicate=True),
    Input('mask-size-slider', 'value'),
    State('mask_list', 'value'),
    prevent_initial_call=True
)
def update_masks_on_resize(scale_factor, selected):
    resized_masks = []
    kernel_size = int(scale_factor * 50)  # Kernel size proportional to the scale factor
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    for mask in ps.masks:
        if scale_factor > 1:
            # Apply dilation
            resized_mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            # Apply erosion
            resized_mask = cv2.erode(mask, kernel, iterations=1)
        
        resized_masks.append(resized_mask)
    
    ps.masks = resized_masks  # Update global mask list
    
    # Update dropdown options to reflect current masks
    options = [{'label': str(i), 'value': i} for i in range(len(resized_masks))]
 
    # Ensure selected values are still valid
    selected = [s for s in selected if s <= len(resized_masks)]

    log("Mask size modulated")
    return options, selected


@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('mask_list', 'options', allow_duplicate=True),
    Output('roi-1', 'value'),
    Output('roi-2', 'value'),
    Output("deg_volcano", "figure", allow_duplicate=True),
    Output("deg_box", "figure", allow_duplicate=True),
    Output("deg_enrich", "figure", allow_duplicate=True),
    Output("deg_enrich2", "figure", allow_duplicate=True),
    Output("deg_celltype", "figure", allow_duplicate=True),
    Output("box", "children", allow_duplicate=True),  
    Input('reset', 'n_clicks'),
    State("alpha-state", "value"),
    prevent_initial_call=True,
)
def reset_button(n_clicks, alpha):
    if n_clicks is None:
        raise PreventUpdate
    
    fig = px.imshow(ps.tsimg_rgb)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))


    ps.masks = None
    ps.boxes = []
    ps.integrated_masks = None
    ps.deg_df = None

    log('Reset.')
    
    return fig, [''], None, None, blank_fig(), blank_fig(), blank_fig(), blank_fig(), blank_fig(), ""



### Everything mode ###

@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('mask_list', 'options', allow_duplicate=True),    
    Input('run_sam', 'n_clicks'),
    State("alpha-state", "value"),
    State("pred_iou_thresh", "value"),
    prevent_initial_call=True,
)
def run_sam_in_everything_mode(n_clicks, alpha, pred_iou_thresh):
    if n_clicks is None:
        raise PreventUpdate

    if not (hasattr(ps, 'adata')):
        log("Error : Data is not selected")
        raise PreventUpdate

    log("Running SAM with Everything mode")
    
    masks = ps.get_mask(pred_iou_thresh = pred_iou_thresh)
    masks_int = ps.integrated_masks
    
    outputimg = np.array(masks_int/len(masks) * 255, dtype = np.uint8)
    im_color = cv2.applyColorMap(outputimg, cv2.COLORMAP_TWILIGHT_SHIFTED)
    blendimg = cv2.addWeighted(ps.tsimg_rgb, 1-alpha, im_color, alpha, 0)

    fig = px.imshow(blendimg)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.update_traces(
        customdata = masks_int,
        hovertemplate="MaskNumber: %{customdata}<extra></extra>")

    mask_names = list(range(1, len(masks)+1))

    log("Running SAM with Everything mode ... Done")
    return fig, mask_names

@app.callback(
    Output('roi-1', 'value', allow_duplicate=True),
    Input('hover_click', 'n_clicks'),
    State('he_image', 'hoverData'),
    State('roi-1', 'value'),
    State("url", "pathname"),
    prevent_initial_call=True
)
def display_click_data(n_clicks, fig, selected, pathname):
    if pathname == "/prompt": # Only for everything mode
        raise PreventUpdate
        
    if n_clicks is None:
        raise PreventUpdate
        
    if len(ps.masks) > 0:
        x, y = fig['points'][0]['x'], fig['points'][0]['y']
        idx = int(ps.integrated_masks[y, x])
        if idx == 0: # Nothing happens when click Mask0(Background)
            raise PreventUpdate
        if not selected or '' in selected:
            selected = [idx]
        elif idx not in selected:
            selected.append(idx)
        elif idx in selected:
            selected.remove(idx)
        return selected


@app.callback(
    Output("he_image","figure", allow_duplicate=True),
    Input('roi-1', 'value'),
    Input('roi-2', 'value'),
    Input("alpha-state", "value"),
    prevent_initial_call=True,
)
def update_selected_mask(roi1, roi2, alpha):
    if roi1 is None or '' in roi1 :
        raise PreventUpdate
    masks = ps.masks
    masks_int = ps.integrated_masks    
    if len(masks) > 0 :
        outputimg = np.array(masks_int/len(masks) * 255, dtype = np.uint8)
        im_color = cv2.applyColorMap(outputimg, cv2.COLORMAP_TWILIGHT_SHIFTED)
        blendimg = cv2.addWeighted(ps.tsimg_rgb, 1-alpha, im_color, alpha, 0)
        
        # Visualize selected mask
        if len(roi1) > 0:

            # Color ROI 1 mask in red
            roi1_mask_int = np.zeros((ps.tsimg_rgb.shape[0], ps.tsimg_rgb.shape[1]), dtype=np.uint8)
            for idx in roi1:
                roi1_mask = np.array(ps.masks[idx-1], dtype = np.uint8)
                roi1_mask_int = np.logical_or(roi1_mask_int, roi1_mask)
            roi1_mask_rgb = cv2.applyColorMap((roi1_mask_int * 255).astype('uint8'), cv2.COLORMAP_OCEAN)
            blendimg = cv2.addWeighted(blendimg, 0.5, roi1_mask_rgb, 0.5, 0)

            # Color ROI 2 mask in blue
            if roi2 is not None and len(roi2) > 0:
                roi2_mask_int = np.zeros((ps.tsimg_rgb.shape[0], ps.tsimg_rgb.shape[1]), dtype=np.uint8)
                for idx in roi2:
                    roi2_mask = np.array(masks[idx-1], dtype = np.uint8)
                    roi2_mask_int = np.logical_or(roi2_mask_int, roi2_mask)
                roi2_mask_rgb = cv2.applyColorMap((roi2_mask_int * 255).astype('uint8'), cv2.COLORMAP_PINK)
                blendimg = cv2.addWeighted(blendimg, 0.5, roi2_mask_rgb, 0.5, 0)
 

    fig = plot_mask(blendimg, masks_int)
    return fig

### Prompt mode ####

@app.callback(
    Output("box", "children", allow_duplicate =True),
    Input('he_image', 'relayoutData'),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def display_relayout_data(relayoutData, pathname):
    if pathname == "/main":
        raise PreventUpdate
    try:
        shapes = relayoutData['shapes']
    
        k = len(shapes)
        x0 = shapes[k-1]['x0']
        y0 = shapes[k-1]['y0']
        x1 = shapes[k-1]['x1']
        y1 = shapes[k-1]['y1']
        
        x0_ = x0 - ps.xrange_[0]
        y0_ = y0 - ps.yrange_[0]
        x1_ = x1 - ps.xrange_[0]
        y1_ = y1 - ps.yrange_[0]
        box = np.array([x0_, y0_, x1_, y1_])

        ps.boxes.append(box)
        print('Box added : {}'.format(box))
        log("Prompt mode : Box added")
        return '# of rectangles : {}'.format(len(ps.boxes))
    
    except:
        raise PreventUpdate


@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('mask_list', 'options', allow_duplicate=True),
    Output('roi-1', 'value', allow_duplicate=True),
    Input('run_sam_prompt', 'n_clicks'),
    State("alpha-state", "value"),
    prevent_initial_call=True,
)
def run_sam_in_prompt_mode(n_clicks, alpha):
    if n_clicks is None:
        raise PreventUpdate
        
    if len(ps.boxes) == 0 :
        log("Error : There is no box prompt")
        raise PreventUpdate

    if not (hasattr(ps, 'adata')):
        log("Error : Data is not selected")
        raise PreventUpdate

    log("Running SAM with Prompt mode")
    
    masks = ps.get_mask_prompt_mode()
    masks_int = ps.integrated_masks
    
    selected = [x for x in range(len(masks))]
    
    outputimg = np.array(masks_int/len(masks) * 255, dtype = np.uint8)
    im_color = cv2.applyColorMap(outputimg, cv2.COLORMAP_TWILIGHT_SHIFTED)
    blendimg = cv2.addWeighted(ps.tsimg_rgb, 1-alpha, im_color, alpha, 0)

    fig = px.imshow(blendimg)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.update_traces(
        customdata = masks_int,
        hovertemplate="MaskNumber: %{customdata}<extra></extra>")
    mask_names = list(range(1, len(masks)+1))

    log("Running SAM with Prompt mode ... Done")
    
    return fig, mask_names, mask_names


### Downstream anylsis ###

@app.callback(
    Output("deg_volcano", "figure", allow_duplicate=True),
    Output("deg_box", "figure", allow_duplicate=True),
    Output("deg_enrich", "figure", allow_duplicate=True),
    Output("deg_enrich2", "figure", allow_duplicate=True),
    Output("deg_celltype", "figure", allow_duplicate=True),
    Output("downstream-loading", "children", allow_duplicate=True),
    Input('run_deg', 'n_clicks'),
    State('roi-1', 'value'),
    State('roi-2', 'value'),
    State('lfc_cutoff', 'value'),
    State('pval_cutoff', 'value'),
    State('geneset', 'value'),
    State('organism-radio', 'value'),
    prevent_initial_call=True
)
def run_downstream_analysis(n_clicks, selected1, selected2, lfc, padj, geneset, organism):

    log("Running Downstream analysis")
    
    if selected1 is None or '' in selected1:
        log("Error: No masks in ROI1")
        raise PreventUpdate
        
    if selected2 is None or '' in selected2 or not selected2:
        selected2 = None

    if len(ps.masks) > 0:

        In_df = ps.extract_degs(selected1, selected2, padj_cutoff = padj, lfc_cutoff = lfc)
        
        try:
            fig_volcano = plot_volcano(In_df)
            fig_box = plot_box(In_df, ps.adata)
        
        except:
            print("Error in DEG")
            fig_volcano = blank_fig()
            fig_box = blank_fig()
            log('Error occurred in Calculating DEG')
        
        try: 
            fig_enrich1 = do_enrichment_analysis_for_ROI1(In_df, geneset, organism)
            fig_enrich2 = do_enrichment_analysis_for_ROI2(In_df, geneset, organism)
        except:
            print("Error in enrichr")
            fig_enrich1 = blank_fig()
            fig_enrich2 = blank_fig()
            log('Error occured in enrichment analysis')
        
        try:
            fig_celltype = plot_deconv_barchart(ps.adata)
        except:
            print("Error in cell deconvolution")
            fig_celltype = blank_fig()
            log('Error occured in plotting cell deconvolution')

        log("Running Downstream analysis ... Done")
    
        return fig_volcano, fig_box, fig_enrich1, fig_enrich2, fig_celltype, ''






###### Export function #####
@app.callback(
    Output("download-barcode", "data"),
    Input("export-barcode", "n_clicks"),
    prevent_initial_call = True,
)
def export_barcode_info(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    try:
        if 'ROIs' in ps.adata.obs.columns:
            celltype_cols = list(ps.adata.obs.columns[ps.adata.obs.columns.str.startswith('celltype')])

            export = ps.adata.obs[celltype_cols + ['ROIs']]
            print("Export ROI data csv")
            return dcc.send_data_frame(export.to_csv, "export.csv")
    except:
        print("Prevent update")
        raise PreventUpdate

@app.callback(
    Output("download-deg", "data"),
    Input("export-deg", "n_clicks"),
    prevent_initial_call = True,
)
def export_deg_table(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    try:
        if ps.deg_df is not None:
            deg = ps.deg_df
            print("Export DEG tables")
            return dcc.send_data_frame(deg.to_csv, "DEG.csv")
    except:
        print("Prevent update")
        raise PreventUpdate




if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Example usage: python app.py --port 9905')
    parser.add_argument('--port', type=int, help='Port number to run the server on')
    args = parser.parse_args()

    if args.port:
        print(f"Starting server on port {args.port}...")
        ps = IAMSAM(config["checkpoint_dir"])
        app.server.run(port=args.port, debug=True, host = '0.0.0.0')
    else:
        print("Please provide a port number to run the server on.")
        print("     Example usage: python app.py --port 9905")





