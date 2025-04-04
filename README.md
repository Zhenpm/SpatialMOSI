# SpatialMOSI
Vertical, Horizontal, and Mosaic Integration of Spatial Omics Data
![image](https://github.com/Zhenpm/SpatialMOSI/blob/main/overviewmosi.jpg)

## Software dependencies

scanpy==1.9.6 <br />
pytorch==1.12.0+cu11.3 <br />
pytorch_geometric==2.4.0 <br />
R==4.2.3 <br />
mclust==5.4.10 <br />

## set up

First clone the repository. 
```
git clone https://github.com/Zhenpm/DisConST.git 
cd DisConST-main
```
Then, we suggest creating a new environment： <br />
```
conda create -n disconst python=3.10 
conda activate disconst
```
Additionally, install the packages required: <br />
```
pip install -r requiements.txt
``` 

## Datasets

We employed seven distinct spatial omics datasets to evaluate model performance:

### 1. Simulated Data
- **Source**: 
  - Single-cell RNA/ATAC from SHARE-seq & SNARE-seq
  - MERFISH mouse brain spatial transcriptomics
- **Content**: 
  - Simulated spatial RNA & ATAC data for mouse cortex
  - Two independent replicates (SHARE/SNARE-based)

### 2. Mouse Brain Spatial ATAC-RNA-seq  
- **Technology**: Spatial ATAC-RNA-seq  
- **Samples**: 
  - Two coronal sections (P21, P22)
  - Paired transcriptome + chromatin accessibility

### 3. Mouse Brain Spatial CUT&Tag-RNA-seq
- **Modalities**:
  - RNA + H3K27ac modification
  - RNA + H3K4me3 modification 

### 4. Human Tonsil Multi-omics
- **Technology**: Spatial-CITE-seq
- **Modalities**: Transcriptome + 20 protein markers

### 5. Mouse Breast Cancer Multi-omics  
- **Technology**: SPOTS
- **Modalities**: Transcriptome + 77 protein markers

### 6. Mouse Embryo Transcriptome
- **Technology**: Stereo-seq
- **Stages**: E9.5, E10.5, E11.5
- **Coverage**: Whole embryo sagittal sections

### 7. Mouse Embryo Chromatin
- **Technology**: Spatial ATAC  
- **Stages**: E12.5, E13.5, E14.5
- **Coverage**: Forebrain regions
Five datasets can be downloaded from https://pan.baidu.com/s/1mmMWKz-GaHqvjTQ-fZ1IZA?pwd=1234
