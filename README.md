# **Data Filtering Networks and Clustering Techniques**

This repository implements **Data Filtering Networks (DFNs)** and **Hierarchical Agglomerative Clustering (HAC)** for dataset optimization and improved machine learning performance.

## **Features**

- **Data Filtering Networks (DFNs)** for high-quality dataset filtering.
- **Hierarchical Agglomerative Clustering (HAC)** using Euclidean distance and average linkage.
- Preprocessing tools for handling missing values, duplicates, and scaling.
- Custom **GridSearchCV** for hyperparameter tuning.
- Evaluation metrics such as Mean Squared Error (MSE), Out-of-Bag (OOB) error, and clustering consistency.

## **Installation**

```bash
git clone https://github.com/Krishna737Sharma/Data-Filtering-Network
cd Data-Filtering-Network
pip install -r requirements.txt

from src.dfn import DataFilteringNetwork

# Initialize and train the DFN model
dfn = DataFilteringNetwork()
dfn.train(dataset)

# Filter unfiltered data
filtered_data = dfn.filter(unfiltered_data)

from src.hac import HierarchicalAgglomerativeClustering

# Initialize and fit HAC model
hac = HierarchicalAgglomerativeClustering(distance_metric="euclidean", linkage="average")
labels = hac.fit_predict(data)
