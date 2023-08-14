# Towards a ‘Resource Cadastre’ for a Circular Economy– Urban-Scale Building Material Detection Using Street View Imagery and Computer Vision

[Deepika Raghu](https://baug.ethz.ch/en/department/people/staff/personen-detail.MjkzOTY1.TGlzdC82NzksLTU1NTc1NDEwMQ==.html), [Martin Juan José Bucher](https://www.mnbucher.com), [Catherine De Wolf](https://www.catherinedewolf.com/about)

[[ Paper ]](https://google.com) – Published in Resources, Conservation & Recycling Advances (RCR Advances) Journal, August 2023

## Table of Contents
1. [Abstract](#abstract)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## Abstract
The lack of data on existing buildings hinders efforts towards repair, reuse, and recycling of materials, which are crucial for mitigating the climate crisis. Manual acquisition of building data is complex and time-consuming, but combining street-level imagery with computer vision could significantly scale-up building materials documentation. We formulate the problem of building facade material detection as a multi-label classification task and present a method using GIS and street view imagery with just a few hundred annotated samples and a fine-tuned image classification model. We make our in the wild dataset publicly available as the Urban Resource Cadastre Repository to encourage future work on automatic building material detection.

![Schematic representation of overall method.](/method.png)

## Dataset
Annotated street-level images from cities like Tokyo, NYC, and Zurich highlighting building facade materials. The classes were distinctively identified for each city based on the materiality of the facades and importance for maintenance and possible end-of-life strategizing, including: ”brick”, ”stucco”, ”rustication”, ”siding”, ”wood”, and ”metal“. A "null” class was added for images without building material information, and an "other” class for materials that were not definable with the given class labels. 

## Model
Three state-of-the-art computer vision models were used for evaluation, namely, Vision Transformer, Swin Transformer V2 and ConvNeXt. Hyperparameters were optimized for each model, and a weighted Binary Cross Entropy (BCE) was used to address class imbalance. Final models were trained using the optimal hyperparameters and three different random seeds. We used a fixed learning rate scheduler with a linear warm-up ratio of 10% and a single-cycle decay based on the cosine function for all models.

We abbreviate the models as follows with the corresponding model name from Huggingface: 

* ViT — Vision Transformer — [google/vit-large-patch16-224](https://huggingface.co/google/vit-large-patch16-224)
* STV2 — Swin Transformer V2 — [microsoft/swinv2-large-patch4-window12-192-22k](https://huggingface.co/microsoft/swinv2-large-patch4-window12-192-22k)
* CNX — ConvNeXt — [facebook/convnext-base-224-22k-1k](https://huggingface.co/facebook/convnext-base-224-22k-1k)

### Results

| Model  | Dataset       | EMR (↑)        | HD (↓)         | Precision (↑)  | Recall (↑)     | Macro F1 (↑)   |
|--------|---------------|----------------|----------------|----------------|----------------|----------------|
| TKY    | ViT           | **0.86** (±0.07) | 0.03 (±0.02) | 0.94 (±0.05) | 0.89 (±0.06) | **0.91** (±0.06) |
|        | STV2          | 0.83 (±0.03)  | **0.03** (±0.01) | 0.93 (±0.03) | **0.90** (±0.05) | 0.90 (±0.03) |
|        | CNX           | 0.82 (±0.03)  | 0.04 (±0.00)  | **0.97** (±0.03) | 0.84 (±0.04) | 0.89 (±0.01) |
|        |               |                |                |                |                |                |
| NYC    | ViT           | 0.79 (±0.08)  | 0.04 (±0.02)  | 0.94 (±0.03) | 0.85 (±0.05) | 0.89 (±0.05) |
|        | STV2          | **0.82** (±0.07) | 0.04 (±0.01) | 0.94 (±0.02) | **0.90** (±0.05) | 0.91 (±0.04) |
|        | CNX           | 0.81 (±0.05)  | **0.03** (±0.01) | **0.95** (±0.01) | 0.89 (±0.05) | **0.91** (±0.03) |
|        |               |                |                |                |                |                |
| ZRH    | ViT           | 0.83 (±0.05)  | 0.04 (±0.01)  | 0.96 (±0.02) | 0.91 (±0.05) | 0.93 (±0.02) |
|        | STV2          | **0.87** (±0.04) | 0.03 (±0.01) | 0.95 (±0.02) | **0.95** (±0.02) | **0.95** (±0.01) |
|        | CNX           | **0.91** (±0.02) | **0.02** (±0.00) | **0.99** (±0.01) | 0.94 (±0.04) | **0.96** (±0.02) |
|        |               |                |                |                |                |                |
| All    | ViT           | 0.82 (±0.07)  | 0.03 (±0.01)  | 0.89 (±0.03) | 0.82 (±0.03) | 0.85 (±0.02) |
|        | STV2          | **0.87** (±0.04) | 0.02 (±0.01) | **0.96** (±0.02) | **0.91** (±0.03) | **0.93** (±0.03) |
|        | CNX           | 0.86 (±0.01)  | 0.02 (±0.00)  | 0.94 (±0.01) | 0.89 (±0.02) | 0.91 (±0.01) |

### Hyperparameters
Hyperparameter tuning was performed on each model to optimize their performance. The final model prediction is a 1-by-N vector with a multi-hot encoding for the different class labels. We obtain this vector by applying a Sigmoid individually on the output logits for each class label and use a Binary Cross Entropy (BCE) loss for the optimization. To tackle the imbalance, we use a **weighted BCE** where we weight the different proportional to the inverse of their training distribution, i.e. more sparse labels are weighted stronger than samples for classes which appear more often. This strategy has shown to be very effective to deal with class imbalance.

In order to find the optimal hyperparameters, we performed a simple grid-search on all three classification models. Specifically, we took the following parameters for the grid-search: 

* LR: Learning Rate: 1e-4, 5e-5, 2e-5
* GAS: Gradient Accumulation Steps: 4, 8, 16
* WD: Weight Decay, 0.01, 0.05, 0.1, ”None”
* RA: *M* for Random Augmentation: 8, 16

For the GAS, we took a fixed batch size of 4 together with the GAS, resulting in a final effective batch size of ```4*GAS```. This was only due to computational constraints, as a bigger batch size lead to out-of-memory issues on CUDA. With GAS, we can approximate a bigger batch size by summing up the gradients over multiple runs and doing backprop after a certain number of fixed steps (i.e. 4 steps with batch size 4 results in a batch size of 16). If ”None” was picked for the Weight Decay, we used vanilla Stochastic Gradient Descent (SGD) for the optimizer. 

For Data Augmentation, we integrated an adopted version of [RandAugment](https://github.com/ildoonet/pytorch-randaugment), where we removed AutoContrast, Invert, Posterize, Solarize, and SolarizeAdd. This was mainly due to the fact that the mentioned augmentation functions alter the color values of the pixels significantly. As our dataset has a much stronger color-prior than a shape-prior (compared to more traditional image classification tasks such as distinguishing between dogs and a bicycle), changing the color values too strongly lead to much weaker performance for the final model. We thus experimented very carefully with data augmentation and mainly relied on geometric augmentation such as rotation, skewing, flipping and other transformations. We used a fixed N=2 and only searched over the range of M={8, 16} for our custom RandAugm implementation

|       | Model             | LR      | GAS | WD   | RA  |
|-------|-------------------|---------|-----|------|-----|
| TKY   | ViT               | 1e-4    | 8   | 0.01 | 16  |
|       | STV2              | 5e-5    | 8   | 0.01 | 16  |
|       | CNX               | 1e-4    | 4   | 0.01 | 8   |
|       |                   |         |     |      |     |
| NYC   | ViT               | 2e-5    | 4   | 0.05 | 16  |
|       | STV2              | 1e-4    | 8   | 0.1  | 16  |
|       | CNX               | 1e-4    | 4   | 0.01 | 8   |
|       |                   |         |     |      |     |
| ZRH   | ViT               | 2e-5    | 4   | 0.05 | 16  |
|       | STV2              | 1e-4    | 16  | 0.05 | 16  |
|       | CNX               | 1e-4    | 4   | 0.05 | 8   |
|       |                   |         |     |      |     |
| All   | ViT               | 1e-4    | 8   | 0.1  | 8   |
|       | STV2              | 1e-4    | 4   | 0.1  | 8   |
|       | CNX               | 1e-4    | 4   | 0.05 | 16  |


## Installation and Setup
```bash
git clone https://github.com/raghudeepika/urban-resource-cadastre-repository.git
cd urban-resource-cadastre-repository

# Optional: create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install git+https://github.com/ildoonet/pytorch-randaugment
```

## Usage

The usage of our source code is quite straightforward. There are two main files you can run to reproduce our results. 

First, you can run ```main_hyperparam.py``` to run the full hyperparameter grid-search for all three models and four datasets. This file will produce a CSV for every dataset and model in the form of ```"results-{dataset}-{m_short}.csv```, where ```dataset``` is replaced by the name of the dataset and ```m_short``` is replaced by the short name (i.e. abbreviation) of the vision model. The full search will take a while depending on your hardware, so running this might take a while.

```bash
.venv/bin/python main_hyperparam.py
```

Second, there is a file ```main_finalmodel.py``` that runs ONE full training loop for your chosen model configuration and dataset. This file is used for the final models that are reported in the paper and you can pick the necessary paramaters from the supplementary paper in the paper or the table with the optimal hyperparameters as presented above. Just open this file in your code editor and edit the corresponding parameters before starting the following script:

```bash
.venv/bin/python main_finalmodel.py
```

Note: The latter will train three distinct models with the same config, but three different random seeds. The final model performance is reported as mean + standard deviation for each corresponding metric.

## Citation
If using our dataset/model, please cite us as follows:
```bibtex
@article{raghu2023resourcecadastre,
  title={Towards a Resource Cadastre for a Circular Economy – Urban-Scale Building Material Detection Using Street View Imagery and Computer Vision},
  author={Raghu, D. and Bucher, M. and DeWolf, C.},
  journal={Resources, Conservation and Recycling},
  year={2023}
}
```

## Acknowledgements


- [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer)
    
- [Swin Transformer V2](https://github.com/ChristophReich1996/Swin-Transformer-V2)

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt-V2)

This research was conducted at the Future Cities Lab Global at ETH Zurich. Future Cities Lab Global is supported and funded by the National Research Foundation, Prime Minister’s Office, Singapore under its Campus for Research Excellence and Technological Enterprise (CREATE) programme and ETH Zurich (ETHZ), with additional contributions from the National University of Singapore (NUS), Nanyang Technological University (NTU), Singapore and the Singapore University of Technology and Design (SUTD). This work is also supported by the Swiss National Science Foundation under grant 200021E_215311.


