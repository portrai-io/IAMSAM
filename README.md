
# **IAMSAM**

<img src="/assets/logo.png"  height="90">

IAMSAM(Image-based Analysis of Molecular signatures using the Segment-Anything Model) is a user-friendly web-based tool designed to analyze spatial transcriptomics data. It utilizes the Segment-anything algorithm to segment H&E images of Visium data and performs statistical analysis to identify differentially expressed genes (DEGs) and their corresponding Gene Ontology (GO) terms for each segmented region.

With its simple and accessible interface, IAMSAM makes it easy for researchers to analyze and interpret their spatial transcriptomics data.

## Requirements

- Python 3.8
- segment-anything (also checkpoint file)
- PyTorch
- OpenCV
- Scanpy
- dash

IAMSAM follows the dependency of 'Segment-anything', which means that it requires the installation of both PyTorch and TorchVision with CUDA support.

## Installation
This is the installation guide for the IAMSAM tool. Follow the steps below to install the tool:

    git clone https://github.com/portrai-io/IAMSAM.git
    conda create -n IAMSAM python=3.8
    conda activate IAMSAM
    pip install -r requirements.txt
    
## User interface
<img src="/assets/ui_example.png" width = 720>


## Usage

1. Start the dash server using the following command:
    
    ```
    CUDA_VISIBLE_DEVICES={GPU_to_use} python app.py --port {port}
    ```
    
2. Open your web browser and go to **`http://localhost:9905`**.
3. Select your Visium data.
4. Click the "Run SAM" button to do segmentation of H&E image.
5. Click the "Run ST analysis" button to perform DEG analysis and GO term enrichment analysis.



## Demo Data

The following table provides information about the datasets that were demonstrated in our paper.

| Sample Name           | Repository                | Download Link                                                                                       |
|-----------------------|---------------------------|-----------------------------------------------------------------------------------------------------|
| Human_Breast_Cancer   | 10X Genomics Dataset      | [Link](https://www.10xgenomics.com/resources/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0)   |
| Mouse_Colon           | Gene Expression Omnibus   | [GSE5213483](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5213483)                         |
| Mouse_Brain_H&E       | Gene Expression Omnibus   | [GSM5519060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5519060)                         |
| Mouse_4T1             | Gene Expression Omnibus   | [GSE196506](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196506)                           |
| Human_Prostate_Cancer | 10X Genomics Dataset      | [Link](https://www.10xgenomics.com/resources/datasets/human-prostate-cancer-adenocarcinoma-with-invasive-carcinoma-ffpe-1-standard-1-3-0) |
| Mouse_Brain_FL        | 10X Genomics Dataset      | [Link](https://www.10xgenomics.com/resources/datasets/adult-mouse-brain-section-1-coronal-stains-dapi-anti-neu-n-1-standard-1-1-0)           |

These datasets were used to demonstrate the capabilities of IAMSAM in our paper. You can access and download the datasets using the provided links.

