# Data Filtering Networks and Clustering Techniques

This repository implements **Data Filtering Networks (DFNs)** and **Hierarchical Agglomerative Clustering (HAC)** for optimizing datasets and enhancing machine learning performance. DFNs are used for efficiently filtering large, uncurated datasets to create high-quality training data for deep learning models. This repository also includes a custom implementation of HAC for clustering experiments and other related techniques.

## Overview of Data Filtering Networks (DFNs)

The core idea behind **Data Filtering Networks** (DFNs) is to learn how to filter a large uncurated dataset to create high-quality training datasets for downstream tasks. In many machine learning applications, the quality of the data is as important as the model’s architecture. DFNs aim to optimize this data selection process by using neural networks to filter out irrelevant or low-quality data, thereby improving model performance.

The research paper demonstrates that using a filtering network trained on a small but high-quality dataset can outperform a network trained on a large, unfiltered dataset. The key insight is that a model trained on a well-filtered dataset can induce state-of-the-art models, even outperforming models trained on much larger datasets.

## Key Contributions of the Research
- **DFNs as a Data Filtering Mechanism**: The research introduces DFNs, neural networks designed to filter data and induce high-quality datasets that improve model performance.
- **Dataset Design**: The paper shows how DFNs can induce large-scale image-text datasets with better performance compared to existing datasets like LAION and OpenAI’s WIT.
- **Compute-Accuracy Tradeoff**: The paper provides evidence that using DFNs can reduce the computational cost of training large models without sacrificing accuracy.
- **Publicly Available High-Quality Datasets**: DFNs can be trained from scratch using only publicly available data, making high-quality dataset creation more accessible.

## Features
- **DFNs** for filtering large, uncurated datasets and creating high-quality training datasets for downstream tasks.
- **HAC** for clustering with Euclidean distance and average linkage.
- **Preprocessing Tools**: Tools for handling missing values, duplicates, scaling data, and dataset splitting.
- **Custom GridSearchCV**: For hyperparameter tuning in regression and clustering tasks.
- **Evaluation Metrics**: Includes Mean Squared Error (MSE), Out-of-Bag (OOB) error, and clustering consistency.

## Installation

Clone this repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
Usage
Running Data Filtering Networks (DFNs)
python
Copy code
from src.dfn import DataFilteringNetwork

dfn = DataFilteringNetwork()
dfn.train(dataset)
filtered_data = dfn.filter(unfiltered_data)
Performing Clustering with HAC
python
Copy code
from src.hac import HierarchicalAgglomerativeClustering

hac = HierarchicalAgglomerativeClustering(distance_metric="euclidean", linkage="average")
labels = hac.fit_predict(data)
Training DFNs with High-Quality Datasets
python
Copy code
# Example: Train a Data Filtering Network using a high-quality dataset
dfn = DataFilteringNetwork()
dfn.train(high_quality_data)
Model Evaluation
After filtering the dataset using DFNs, evaluate the performance of models trained on filtered data against those trained on unfiltered datasets using metrics like MSE, OOB error, and clustering consistency.

Datasets Used
UCI Vehicle Silhouettes Dataset: Used for clustering experiments.
Online News Popularity Dataset: Used for regression tasks.
Diabetes Dataset: Used for Random Forest regression experiments.
High-Quality Image-Text Datasets: Induced using DFNs for training models like CLIP.
Results
The paper demonstrates the use of DFNs to filter and curate large-scale datasets. The DFN-2B and DFN-5B datasets outperformed existing models, with notable improvements in zero-shot accuracy on ImageNet and better robustness to distribution shifts.

Key Findings:
DFNs can outperform models trained on much larger datasets (e.g., LAION-2B) when used to curate high-quality datasets.
Models trained on DFN-2B achieved state-of-the-art results on a variety of benchmarks, including ImageNet and Visual Question Answering (VQA).
DFNs improve computational efficiency by enabling smaller models to outperform larger models trained on unfiltered datasets.
License
This project is licensed under the MIT License. See the LICENSE file for details.

References
Fang et al., Data Filtering Networks, arXiv 2023.
LAION: Large-scale, open datasets for contrastive learning.
DataComp Benchmark: Provides a framework for dataset evaluation and comparison.
