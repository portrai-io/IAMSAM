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



# Load Configuration
config = load_json('config/config.json')

# Load data from 'data' folder
data_files = os.listdir('data/')
data_files.sort()
flists = []
for sample in data_files:
    if check_sample_folder(os.path.join('data/', sample)):
        flists.append({'label' : sample, 'value' : 'data/{}'.format(sample)})

    
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

@app.callback(Output("page-content", "children"), 
              [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/prompt":
        return prompt.create_layout(app, flists, config)
    else:
        return everything.create_layout(app, flists, config),

@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('overlay_dropdown', 'value', allow_duplicate=True),
    Output("reset_load", "n_clicks"),
    Input('he_dropdown', 'value'), 
    State("reset_load", "n_clicks"),
    prevent_initial_call=True
)
def load_he_image(tissue_dir, reset):
    if tissue_dir is None:
        raise PreventUpdate
    ps.adata = None
    ps.load_visium(tissue_dir)
    fig = px.imshow(ps.tsimg_rgb)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    
    ps.boxes = []
    
    return fig, [''], reset+1


@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('overlay_dropdown', 'options'),    
    Input('run_sam', 'n_clicks'),
    State("alpha-state", "value"),
    State("pred_iou_thresh", "value"),
    prevent_initial_call=True,
)
def run_sam_in_everything_mode(n_clicks, alpha, pred_iou_thresh):
    if n_clicks is None:
        raise PreventUpdate
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
    return fig, mask_names


@app.callback(
    Output("he_image","figure", allow_duplicate=True),
    Input('overlay_dropdown', 'value'),
    Input("alpha-state", "value"),
    prevent_initial_call=True,
 )
def update_selected_mask(selected, alpha):
    if selected is None or '' in selected :
        raise PreventUpdate
    masks = ps.masks
    masks_int = ps.integrated_masks    
    if len(masks) > 0 :
        outputimg = np.array(masks_int/len(masks) * 255, dtype = np.uint8)
        im_color = cv2.applyColorMap(outputimg, cv2.COLORMAP_TWILIGHT_SHIFTED)
        blendimg = cv2.addWeighted(ps.tsimg_rgb, 1-alpha, im_color, alpha, 0)
        
        # Visualize selected mask
        if len(selected) > 0:          
            selmask = np.zeros((ps.tsimg_rgb.shape[0], ps.tsimg_rgb.shape[1]))
            for idx in selected:
                selmask = selmask + masks[idx-1]
            selmask = np.array(selmask > 0)

            selmask_int = np.array(selmask * 255, dtype = np.uint8)
            selmask_bgr = cv2.applyColorMap(selmask_int, cv2.COLORMAP_OCEAN)
            blendimg = cv2.addWeighted(blendimg, 0.5, selmask_bgr, 0.5, 0)

    fig = plot_mask(blendimg, masks_int)
    return fig


@app.callback(
    Output("deg_volcano", "figure", allow_duplicate=True),
    Output("deg_box", "figure", allow_duplicate=True),
    Output("deg_enrich", "figure", allow_duplicate=True),
    Output("deg_celltype", "figure", allow_duplicate=True),
    Output("msg", "children", allow_duplicate = True),
    Input('run_deg', 'n_clicks'),
    State('overlay_dropdown', 'value'),
    State('lfc_cutoff', 'value'),
    State('pval_cutoff', 'value'),
    State('geneset', 'value'),
    State('model_dropdown', 'value'),
    State("organism-radio", 'value'),
    prevent_initial_call=True
)
def run_downstream_analysis(n_clicks, selected, lfc, padj, geneset, model, organism):
    if selected is None or '' in selected :
        raise PreventUpdate
        
    if len(ps.masks) > 0:
        In_df = ps.extract_degs(selected, padj_cutoff = padj, lfc_cutoff = lfc)
        
        msg = 'Analysis Done.'
        try:
            fig_volcano = plot_volcano(In_df)
            fig_box = plot_box(In_df, ps.adata)
        
        except:
            print("Error in DEG")
            fig_volcano = blank_fig()
            fig_box = blank_fig()
            msg = 'Error occurred'
        
        try: 
            fig_enrich = do_enrichment_analysis(In_df, geneset, organism)
        except:
            print("Error in enrichr")
            fig_enrich = blank_fig()
            msg = 'Error occured'
        
        try:
            fig_celltype = do_celltypist(model, ps.adata, organism)
        except:
            print("Error in celltypist")
            fig_celltype = blank_fig()
            msg = 'Error occured'
        
        return fig_volcano, fig_box, fig_enrich, fig_celltype, msg


@app.callback(
    Output('overlay_dropdown', 'value', allow_duplicate=True),
    Input('hover_click', 'n_clicks'),
    State('he_image', 'hoverData'),
    State('overlay_dropdown', 'value'),
    State("url", "pathname"),
    prevent_initial_call=True
)
def display_click_data(n_clicks, fig, selected, pathname):
    if pathname == "/IAMSAM/prompt":
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
    Output("box", "children", allow_duplicate =True),
    Input('he_image', 'relayoutData'),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def display_relayout_data(relayoutData, pathname):
    if pathname == "/IAMSAM/main":
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

        return '# of rectangles : {}'.format(len(ps.boxes))
    
    except:
        raise PreventUpdate


@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('overlay_dropdown', 'value', allow_duplicate=True),
    Input('run_sam_prompt', 'n_clicks'),
    State("alpha-state", "value"),
    prevent_initial_call=True,
)
def run_sam_in_prompt_mode(n_clicks, alpha):
    if n_clicks is None:
        raise PreventUpdate
        
    if len(ps.boxes) == 0 :        
        raise PreventUpdate
        
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
    return fig, selected


@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('overlay_dropdown', 'value', allow_duplicate=True),
    Output("deg_volcano", "figure", allow_duplicate=True),
    Output("deg_box", "figure", allow_duplicate=True),
    Output("deg_enrich", "figure", allow_duplicate=True),
    Output("deg_celltype", "figure", allow_duplicate=True),
    Output("msg", "children", allow_duplicate = True),
    Input('reset', 'n_clicks'),
    Input('reset_load', 'n_clicks'),
    State("alpha-state", "value"),
    prevent_initial_call=True,
)
def reset_button_in_everything_mode(n_clicks, reset_load, alpha):
    if n_clicks is None:
        raise PreventUpdate
    if reset_load is None:
        raise PreventUpdate
        
    fig = px.imshow(ps.tsimg_rgb)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))

    return fig, [''], blank_fig(), blank_fig(), blank_fig(), blank_fig(), None


@app.callback(
    Output("he_image", "figure", allow_duplicate=True),
    Output('overlay_dropdown', 'value', allow_duplicate=True),
    Output("deg_volcano", "figure", allow_duplicate=True),
    Output("deg_box", "figure", allow_duplicate=True),
    Output("deg_enrich", "figure", allow_duplicate=True),
    Output("deg_celltype", "figure", allow_duplicate=True),
    Output("msg", "children", allow_duplicate = True),
    Output("box", "children", allow_duplicate =True),
    Input('reset_prompt', 'n_clicks'),
    Input('reset_load', 'n_clicks'),
    prevent_initial_call=True,
)
def reset_button_in_prompt_mode(n_clicks, reset_load):
    if n_clicks is None:
        raise PreventUpdate

    if reset_load is None:
        raise PreventUpdate
    
    fig = px.imshow(ps.tsimg_rgb)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    ps.boxes = []
    return fig, [''], blank_fig(), blank_fig(), blank_fig(), blank_fig(), None, None


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
    Output("download-barcode", "data"),
    Input("export-barcode", "n_clicks"),
    prevent_initial_call = True,
)
def export_barcode_info(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    try:
        if 'mask_in' in ps.adata.obs.columns:
            export = pd.DataFrame(ps.adata.obs['mask_in'])
            print("Export barcode informations")
            return dcc.send_data_frame(export.to_csv, "mask_in.csv")
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
        if ps.In_df is not None:
            deg = ps.In_df
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





