# Urban Resource Cadastre Repository
Urban Resource Cadastre Repository: Multi-label classification model and annotated street-level imagery dataset for building facade material detection. Curated from cities including Tokyo, NYC, and Zurich.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## Introduction
The lack of data on existing buildings hinders efforts towards repair, reuse, and recycling of materials, which are crucial for mitigating the climate crisis. Manual acquisition of building data is complex and time-consuming, but combining street-level imagery with computer vision could significantly scale-up building materials documentation. We formulate the problem of building facade material detection as a multi-label classification task and present a method using GIS and street view imagery with just a few hundred annotated samples and a fine-tuned image classification model. We make our in the wild dataset publicly available as the Urban Resource Cadastre Repository to encourage future work on automatic building material detection.

![Schematic representation of overall method.](/method.png)

## Dataset
Annotated street-level images from cities like Tokyo, NYC, and Zurich highlighting building facade materials. The classes were distinctively identified for each city based on the materiality of the facades and importance for maintenance and possible end-of-life strategizing, including: brick, stucco, rustication, siding, wood, and Metal. A "null” class was added for images without building material information, and an "other” class for materials that were not defined with the given class labels. 

## Model
Three state-of-the-art computer vision models were used for evaluation, namely, Vision Transformer, Swin Transformer V2 and ConvNeXt. Hyperparameters were optimized for each model, and a weighted Binary Cross Entropy (BCE) was used to address class imbalance. Final models were trained using the optimal hyperparameters and three different random seeds. We used a fixed learning rate scheduler with a linear warm-up ratio of 10% and a single-cycle decay based on the cosine function for all models.

### Performance
Our model showcases:
- Tokyo: F1 Score - 0.91 
- NYC: F1 Score - 0.91
- Zurich: F1 Score - 0.96
- Merged dataset: F1 Score - 0.93

## Installation and Setup
```bash
git clone https://github.com/raghudeepika/urban-resource-cadastre-repository.git
cd urban-resource-cadastre-repository

# Install dependencies if you have any
pip install -r requirements.txt
```

## Usage

Step-by-step instructions for utilizing model:

## Citation
If using our dataset/model, please cite:
```bibtex
@article{raghu2023resourcecadastre,
  title={Towards a Resource Cadastre for a Circular Economy – Urban-Scale Building Material Detection Using Street View Imagery and Computer Vision},
  author={Raghu, D. and Bucher, M. and DeWolf, C.},
  journal={Resources, Conservation and Recycling},
  year={2023}
}
```

## Acknowledgements

- [**Vision Transformer (ViT)**](https://github.com/google-research/vision_transformer)
    
- [**Swin Transformer**](https://github.com/microsoft/Swin-Transformer](https://github.com/ChristophReich1996/Swin-Transformer-V2)

- [**ConvNeXt-V2**](https://github.com/microsoft/Swin-Transformer(https://github.com/facebookresearch/ConvNeXt-V2)

This research was conducted at the Future Cities Lab Global at ETH Zurich. Future Cities Lab Global is supported and funded by the National Research Foundation, Prime Minister’s Office, Singapore under its Campus for Research Excellence and Technological Enterprise (CREATE) programme and ETH Zurich (ETHZ), with additional contributions from the National University of Singapore (NUS), Nanyang Technological University (NTU), Singapore and the Singapore University of Technology and Design (SUTD). This work is also supported by the Swiss National Science Foundation under grant 200021E_215311.


