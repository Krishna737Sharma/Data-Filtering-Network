# Data Filtering Networks and Clustering Techniques  

This repository provides an implementation of Data Filtering Networks (DFNs) and clustering techniques for optimizing datasets and enhancing machine learning model performance. It includes tools for hierarchical agglomerative clustering (HAC), preprocessing, and evaluation on benchmark datasets.  

## Features  
- **Data Filtering Networks (DFNs)**: Efficient dataset filtering to improve training performance for tasks like Contrastive Language-Image Pre-training (CLIP).  
- **Hierarchical Agglomerative Clustering (HAC)**: Custom implementation of HAC for clustering datasets with Euclidean distance and average linkage.  
- **Preprocessing Tools**: Handling duplicates, missing values, scaling, and dataset splitting.  
- **Custom GridSearchCV**: For hyperparameter tuning in regression and clustering tasks.  
- **Evaluation Methods**: Tools for calculating Mean Squared Error (MSE), Out-of-Bag (OOB) error, and clustering consistency metrics.  

## Contents  
- `src/`: Source code for DFNs, HAC, and evaluation scripts.  
- `data/`: Placeholder for datasets or scripts to fetch datasets.  
- `notebooks/`: Jupyter notebooks for experimentation and visualization.  
- `results/`: Folder to store logs, metrics, and plots.  

## Installation  
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

from src.dfn import DataFilteringNetwork
dfn = DataFilteringNetwork()
dfn.train(dataset)
filtered_data = dfn.filter(unfiltered_data)

from src.hac import HierarchicalAgglomerativeClustering
hac = HierarchicalAgglomerativeClustering(distance_metric="euclidean", linkage="average")
labels = hac.fit_predict(data)

Datasets
UCI Vehicle Silhouettes Dataset: Used for clustering experiments.
Online News Popularity Dataset: Used for regression tasks with Support Vector Machines (SVM).
Diabetes Dataset: Used for Random Forest regression experiments.
Results
Improved dataset quality using DFNs, leading to better model accuracy and robustness.
Comparison of custom implementations with scikit-learn models for clustering, regression, and filtering tasks.
References
Fang et al., Data Filtering Networks, arXiv 2023.
License
This repository is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.
