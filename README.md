# <img src="/assets/logo.png"  height="90">
 

[![GitHub license](https://img.shields.io/github/license/portrai-io/IAMSAM)](https://github.com/portrai-io/IAMSAM/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/portrai-io/IAMSAM)](https://github.com/portrai-io/IAMSAM/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/portrai-io/IAMSAM)](https://github.com/portrai-io/IAMSAM/issues)
[![GitHub contributors](https://img.shields.io/github/contributors/portrai-io/IAMSAM)](https://github.com/portrai-io/IAMSAM/graphs/contributors)

IAMSAM (Image-based Analysis of Molecular signatures using the Segment-Anything Model) is a user-friendly web-based tool designed to analyze ST data. This repository contains the code and resources to utilize the functionalities of IAMSAM described in [our paper](https://doi.org/10.1101/2023.05.25.542052).


## Features
IAMSAM utilizes the [Segment-Anything Model](https://github.com/facebookresearch/segment-anything) for H&E image segmentation, which allows for morphological guidance in selecting ROIs for users. IAMSAM offers users with two modes for running the SAM algorithm: everything-mode and prompt-mode.

- **Everything-mode** : An automatic mode that generates masks for the entire image, providing a comprehensive analysis of the spatial gene expression patterns.

- **Prompt-mode** : An interactive mode that allows users to guide the segmentation process by providing box prompts.

After selecting ROIs, IAMSAM automatically performs downstream analysis including identification of differentially expressed genes, enrichment analysis, and cell type prediction within the selected regions.

## Tutorial Video
[![Tutorial Video](/assets/title.png)](https://youtu.be/ri1OB4W210Q)



## Requirements

- Python 3.8
- segment-anything (also checkpoint file)
- PyTorch
- OpenCV
- Scanpy
- dash

IAMSAM follows the dependency of **Segment-anything**, which means that it requires the installation of both PyTorch and TorchVision with CUDA support.

To use SAM, you need to download the **ViT-H SAM model file** (`sam_vit_h_4b8939.pth`) from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) and then place it in the `config` folder.

## Installation
This is the installation guide for the IAMSAM tool. Follow the steps below to install the tool:

    git clone https://github.com/portrai-io/IAMSAM.git
    conda create -n IAMSAM python=3.8
    conda activate IAMSAM
    pip install -r requirements.txt
    


## Usage

1. Start the dash server using the following command:
    
    ```
    python app.py --port {port_to_use}
    ```
    
2. Open your web browser and go to **`http://localhost:{port}`**.
3. Place your ST data in `data` folder. Please refer to this [guide](https://github.com/portrai-io/IAMSAM/blob/main/data/rule.md)
4. Click the "Run SAM" button to do segmentation of H&E image.
5. Click the "Run ST analysis" button to perform DEG analysis and GO term enrichment analysis.

## Demo Data

| Sample Name           | Repository                | Download Link                                                                                       |
|-----------------------|---------------------------|-----------------------------------------------------------------------------------------------------|
| Human_Breast_Cancer   | 10X Genomics Dataset      | [Link](https://www.10xgenomics.com/resources/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0)   |
| Mouse_Colon           | Gene Expression Omnibus   | [GSE5213483](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5213483)                         |
| Mouse_Brain_H&E       | Gene Expression Omnibus   | [GSM5519060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5519060)                         |
| Mouse_4T1             | Gene Expression Omnibus   | [GSE196506](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196506)                           |
| Human_Prostate_Cancer | 10X Genomics Dataset      | [Link](https://www.10xgenomics.com/resources/datasets/human-prostate-cancer-adenocarcinoma-with-invasive-carcinoma-ffpe-1-standard-1-3-0) |
| Mouse_Brain_FL        | 10X Genomics Dataset      | [Link](https://www.10xgenomics.com/resources/datasets/adult-mouse-brain-section-1-coronal-stains-dapi-anti-neu-n-1-standard-1-1-0)           |

These datasets were used to demonstrate the capabilities of IAMSAM in our paper. You can access and download the datasets using the provided links.

## Citation
If you find IAMSAM helpful for your work, consider citing [our paper](https://doi.org/10.1101/2023.05.25.542052)


## Contact
For any questions or inquiries, please contact us at [contact@portrai.io](mailto:contact@portrai.io).


