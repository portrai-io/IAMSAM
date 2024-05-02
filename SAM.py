import numpy as np
import scanpy as sc
import pandas as pd
import cv2
import json
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
import gseapy
from gseapy import barplot, dotplot
from utils import *

class IAMSAM():
    def __init__(self, chkp_path, model_type = 'vit_h'):
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if DEVICE.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
        
        sam = sam_model_registry[model_type](checkpoint=chkp_path)
        sam.to(device=DEVICE)
        
        self.predictor = SamPredictor(sam_model=sam)
        
        print("\n\nIAMSAM loaded.")
        self.sam = sam
        self.masks = []
        self.boxes = []
    
    def load_visium(self, h5ad_dir):

        # Load Anndata
        self.adata = sc.read_h5ad(h5ad_dir) 
        self.adata.var_names_make_unique()

        self.adata.X = self.adata.layers['counts']
        sc.pp.normalize_total(self.adata, target_sum = 1e4)
        sc.pp.log1p(self.adata)
       
        library_id = list(self.adata.uns['spatial'].keys())[0]

        # Tissue image(Before crop)

        self.tsimg_rgb = cv2.convertScaleAbs(self.adata.uns['spatial'][library_id]['images']['hires']*255)
        self.tsimg_bgr = cv2.cvtColor(self.tsimg_rgb, cv2.COLOR_RGB2BGR)
        
        tissue_hires_scalef = self.adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
        
        self.adata.obs[['imgcol', 'imgrow']] = self.adata.obsm['spatial']
        self.adata.obs['imgrow_'] = tissue_hires_scalef * self.adata.obs['imgrow']
        self.adata.obs['imgcol_'] = tissue_hires_scalef * self.adata.obs['imgcol']
        
        self.xrange = [np.min(self.adata.obs['imgcol_']), np.max(self.adata.obs['imgcol_'])] 
        self.yrange = [np.min(self.adata.obs['imgrow_']), np.max(self.adata.obs['imgrow_'])] 

        pad = 0.5
        self.xrange_ = [round(self.xrange[0]-pad), round(self.xrange[1]+pad)]
        self.yrange_ = [round(self.yrange[0]-pad), round(self.yrange[1]+pad)]

        # Tissue image (After crop)
        self.tsimg_rgb_cropped = self.tsimg_rgb[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]]
        self.tsimg_bgr_cropped = self.tsimg_bgr[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]]
        
        print("Image Loaded.")
        print('X range : {} ~ {} '.format(self.xrange[0], self.xrange[1]))
        print('Y range : {} ~ {} '.format(self.yrange[0], self.yrange[1]))

        
        # For Prompt mode
        self.predictor.set_image(self.tsimg_rgb_cropped)        
        self.prompt_flag = False
    
    
    def get_mask_prompt_mode(self):

        input_boxes = torch.tensor(self.boxes, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, self.tsimg_rgb_cropped.shape[:2])

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        mask_list = []
        for mask in masks:  
            mask_ = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1])) # Mask size should be original size
            mask_[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]] = mask[0].cpu().numpy()
            mask_list.append(mask_)
        self.masks = mask_list
        self.masks_backup = mask_list
        
        masks_integrated = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
        for ii, mm in enumerate(self.masks):
            masks_integrated[mm == 1] = ii # 0-based index    
        self.integrated_masks = masks_integrated
        return self.masks
    
    def get_mask(self, 
                 box = None,
                points_per_side=32,
                pred_iou_thresh=0.95, #KEY PARAM : MORE VALUE, LESS CLUSTERS
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100  # Requires open-cv to run post-processing
                ):

        mask_generator = SamAutomaticMaskGenerator(
                        model=self.sam,
                        points_per_side= points_per_side,
                        pred_iou_thresh=pred_iou_thresh, # KEY PARAM : MORE VALUE, LESS CLUSTERS
                        stability_score_thresh=stability_score_thresh,
                        crop_n_layers=crop_n_layers,
                        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                        min_mask_region_area=min_mask_region_area  # Requires open-cv to run post-processing
                        )

        sam_result = mask_generator.generate(self.tsimg_rgb_cropped)
        masks_cropped = [mask['segmentation'] for mask in sam_result] # cropped size masks
        n_total_masks = len(masks_cropped)
        
        # Get original masks from cropped mask
        masks = []
        for mask_ in masks_cropped:
            mask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
            mask[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]] = mask_
            masks.append(mask) # original size
    
        # Set on_tissue mask 
        pixels = np.column_stack((self.adata.obs['imgrow_'].values.astype(int), 
                                  self.adata.obs['imgcol_'].values.astype(int)))
        tissue_mask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
        tissue_mask[pixels[:, 0], pixels[:, 1]] = 1

        # filtering out sam_result values that are not on the tissue 
        on_tissue_sam_result = []
        for idx, m in enumerate(masks):
            prop = np.sum(cv2.bitwise_and(m, tissue_mask)) / self.adata.n_obs
            
            if prop > 0.001:
                on_tissue_sam_result.append(sam_result[idx])
        on_tissue_masks = [mask['segmentation'] for mask in sorted(on_tissue_sam_result, key=lambda x: x['area'], reverse=True)]  # cropped size
        
        # Get original mask of on_tissue_masks
        masks = []
        for mask_ in on_tissue_masks:
            mask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
            mask[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]] = mask_
            masks.append(mask)
        self.masks = masks
        self.masks_backup = masks
        
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(on_tissue_sam_result)
        self.annotated_image = mask_annotator.annotate(self.tsimg_bgr_cropped, detections)
        
        print("{} masks detected, after excluding {} masks not on the tissue".format(
            len(on_tissue_masks), n_total_masks - len(on_tissue_masks)))
        
        # Save integrated mask
        masks_integrated = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
        for ii , mm in enumerate(self.masks):
            masks_integrated[mm == 1] = ii + 1 # 1-based index    
        self.integrated_masks = masks_integrated

        
        
        return self.masks

    
    def extract_degs(self, selected1, selected2, padj_cutoff, lfc_cutoff):

        # Add selected mask as 'ROI-1' in adata.obs
        selmask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
        for idx in selected1:
            selmask = selmask + self.masks[idx]
        selmask = np.array(selmask > 0)

        roi1 = []
        for idx in range(self.adata.n_obs):
            iiy = round(self.adata.obs.iloc[idx,:]['imgrow_'])
            iix = round(self.adata.obs.iloc[idx,:]['imgcol_'])
            roi1.append(selmask[iiy, iix])

        # if ROI-2 exists then add ROI-2, else add others.
        if not selected2:
            selmask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
            for idx in selected2:
                selmask = selmask + self.masks[idx]
            selmask = np.array(selmask > 0)
    
            roi2 = []
            for idx in range(self.adata.n_obs):
                iiy = round(self.adata.obs.iloc[idx,:]['imgrow_'])
                iix = round(self.adata.obs.iloc[idx,:]['imgcol_'])
                roi2.append(selmask[iiy, iix])
            
        else:
            roi2 = np.invert(roi1)
            

        self.adata.obs['ROIs'] = ['ROI1' if in1 else 'ROI2' if in2 else '' for in1, in2 in zip(roi1, roi2)]
        
        # Test DEG
        adata_roi = self.adata[np.isin(self.adata.obs['ROIs'], ['ROI1', 'ROI2']),:].copy()
        sc.tl.rank_genes_groups(adata_roi, 'ROIs', method='wilcoxon', key_added='DEG')

        # DEG_result
        self.deg_df = sc.get.rank_genes_groups_df(adata_roi, group = 'ROI1', key = 'DEG')
        self.deg_df['-log10Padj'] = -np.log10(self.deg_df['pvals_adj'])
        
        self.deg_df['DE'] = 'None'
        self.deg_df.loc[(self.deg_df.pvals_adj < float(padj_cutoff)) & (self.deg_df.logfoldchanges > float(lfc_cutoff)), 'DE'] = 'ROI1'
        self.deg_df.loc[(self.deg_df.pvals_adj < float(padj_cutoff)) & (self.deg_df.logfoldchanges < -float(lfc_cutoff)), 'DE'] = 'ROI2'
            
        print("Extract DEGs")
        return self.deg_df
