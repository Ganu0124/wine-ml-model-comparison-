ganu023334
# 🍷 Comparative Analysis of Machine Learning Algorithms Using Wine Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Plotly-Animated%20&%20Interactive-success.svg" alt="Plotly">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 🚀 Project Overview
This project delivers a comprehensive comparative analysis of multiple machine learning algorithms applied to the classic **Wine Dataset** sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine). 

Taking visualization a step further, this pipeline features **fully animated and interactive Plotly dashboards**. It includes categorical ranking bar charts and **time-stepped clustering comparisons with built-in play/pause controls** for exploring K-Means, DBSCAN, and Hierarchical grouping dynamics.

---

## 🤖 Algorithms Evaluated
* **Linear Models:** Linear Regression, Logistic Regression, Ridge, Lasso, Perceptron
* **Instance-Based & Probabilistic:** K-Nearest Neighbors (KNN), Naive Bayes (GaussianNB)
* **Tree-Based & Ensembles:** Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Extra Trees
* **Support Vector Machines:** SVM (Linear), SVM (RBF)
* **Neural Networks:** Multi-Layer Perceptron (MLPClassifier)

---

## 🛠️ Techniques & Methods Used
* **Data Preprocessing:** Feature Scaling (`StandardScaler`), Target Label Encoding
* **Dimensionality Reduction:** Principal Component Analysis (`PCA`)
* **Unsupervised Clustering:** K-Means (with Elbow Method), DBSCAN, Agglomerative Hierarchical Clustering *(featuring animated algorithm switching)*
* **Evaluation & Diagnostics:** Confusion Matrices, Feature Importance Rankings, Loss Curves, Animated Categorical Performance Metrics

---

## 🏆 Key Results
* **Random Forest** and **Extra Trees** classifiers achieved exceptional classification performance (**100% accuracy**) on the test split.
* Ensemble methods vastly outperformed simpler linear approaches on this specific multi-class chemical feature set.

---

## 📦 Libraries Required
Ensure your environment includes the updated animation-compatible libraries:
* `scikit-learn`
* `pandas`
* `numpy`
* `plotly`
* `ucimlrepo`

---

## ⚙️ Run Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ganu0124/wine-ml-model-comparison-.git](https://github.com/Ganu0124/wine-ml-model-comparison-.git)
   cd wine-ml-model-comparison-
