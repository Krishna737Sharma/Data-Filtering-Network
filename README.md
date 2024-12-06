# Data Filtering Networks and Clustering Techniques

This repository focuses on implementing **Data Filtering Networks (DFNs)** and **Hierarchical Agglomerative Clustering (HAC)** for dataset optimization and improved machine learning performance.  

## Features
- **DFNs** for high-quality dataset filtering.
- **HAC** with Euclidean distance and average linkage.
- Preprocessing tools for handling missing values, duplicates, and scaling.
- Custom **GridSearchCV** for hyperparameter tuning.
- Evaluation metrics like Mean Squared Error (MSE), Out-of-Bag (OOB) error, and clustering consistency.

## Installation
```bash
git clone https://github.com/Krishna737Sharma/Data-Filtering-Network
cd Data-Filtering-Network
pip install -r requirements.txt




## Usage
```python
# Running Data Filtering Networks (DFNs)
from src.dfn import DataFilteringNetwork

dfn = DataFilteringNetwork()
dfn.train(dataset)
filtered_data = dfn.filter(unfiltered_data)

# Performing Clustering with HAC
from src.hac import HierarchicalAgglomerativeClustering

hac = HierarchicalAgglomerativeClustering(distance_metric="euclidean", linkage="average")
labels = hac.fit_predict(data)

# Datasets Used

- **UCI Vehicle Silhouettes Dataset**: Used for clustering experiments.
- **Online News Popularity Dataset**: Used for regression tasks.
- **Diabetes Dataset**: Used for Random Forest regression experiments.

# License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

# References

- Fang et al., *Data Filtering Networks*, arXiv 2023.

# Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
