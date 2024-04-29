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

        # Basic QC
        sc.pp.filter_genes(self.adata, min_counts = 5)
        
        ### TO-DO: Check normalized ### 
        sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(self.adata)
        
        print("Anndata prepared.")
        
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
        #pad = (self.xrange[1]-self.xrange[0])/(np.max(self.adata.obs['array_col']) - np.min(self.adata.obs['array_col']))
        self.xrange_ = [round(self.xrange[0]-pad), round(self.xrange[1]+pad)]
        self.yrange_ = [round(self.yrange[0]-pad), round(self.yrange[1]+pad)]

        # Tissue image (After crop)
        self.tsimg_rgb_cropped = self.tsimg_rgb[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]]
        self.tsimg_bgr_cropped = self.tsimg_bgr[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]]
        
        print("Image Loaded.")
        #plt.imshow(self.tsimg_bgr_)
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
        
        masks_integrated = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
        for ii, mm in enumerate(self.masks):
            masks_integrated[mm == 1] = ii + 1 # 1-based index    
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
        tissue_mask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
        for idx in range(self.adata.n_obs):
            iiy = round(self.adata.obs.iloc[idx,:]['imgrow_'])
            iix = round(self.adata.obs.iloc[idx,:]['imgcol_'])
            tissue_mask[iiy, iix] = 1
                
        # filtering out sam_result values that are not on the tissue 
        on_tissue_sam_result = []
        for idx, m in enumerate(masks):
            prop = np.sum(cv2.bitwise_and(m, tissue_mask)) / np.sum(m) * 100
            
            if prop > 0.01:
                on_tissue_sam_result.append(sam_result[idx])
        on_tissue_masks = [mask['segmentation'] for mask in sorted(on_tissue_sam_result, key=lambda x: x['area'], reverse=True)]  # cropped size
        
        # Get original mask of on_tissue_masks
        masks = []
        for mask_ in on_tissue_masks:
            mask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
            mask[self.yrange_[0]:self.yrange_[1], self.xrange_[0]:self.xrange_[1]] = mask_
            masks.append(mask)
        self.masks = masks
        
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

    
    def extract_degs(self, selected, padj_cutoff, lfc_cutoff):
        
        # Add selected mask as 'In' in adata.obs
        selmask = np.zeros((self.tsimg_rgb.shape[0], self.tsimg_rgb.shape[1]))
        for idx in selected:
            selmask = selmask + self.masks[idx]
        selmask = np.array(selmask > 0)

        mask_in = []
        for idx in range(self.adata.n_obs):
            iiy = round(self.adata.obs.iloc[idx,:]['imgrow_'])
            iix = round(self.adata.obs.iloc[idx,:]['imgcol_'])
            mask_in.append(selmask[iiy, iix])
        
        nameidx = ['Out', 'In']
        self.adata.obs['mask_in'] = [nameidx[m] for m in mask_in] # To String 
        
        # Test DEG
        sc.tl.rank_genes_groups(self.adata, 'mask_in', method='wilcoxon', key_added='DEG')
        self.In_df = sc.get.rank_genes_groups_df(self.adata, group = 'In', key = 'DEG')
        self.In_df['-log10Padj'] = -np.log10(self.In_df['pvals_adj'])
        
        self.In_df['DE'] = False
        self.In_df.loc[(self.In_df.pvals_adj < float(padj_cutoff)) & (abs(self.In_df.logfoldchanges) > float(lfc_cutoff)), 'DE'] = True
            
        print("Extract DEGs")
        
        return self.In_df