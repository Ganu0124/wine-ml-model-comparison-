# Comparative Analysis of Machine Learning Algorithms Using Wine Dataset

## Project Overview
This project compares multiple machine learning algorithms using the Wine Dataset from the UCI repository.

## Algorithms Used
- Logistic Regression
- KNN
- Decision Tree
- Naive Bayes
- SVM
- Random Forest
- AdaBoost
- Gradient Boosting
- Extra Trees
- Neural Network
- Perceptron

## Techniques Used
- Feature Scaling
- PCA
- Clustering
- Confusion Matrix
- Feature Importance
- Data Visualization

## Best Model
Random Forest and Extra Trees achieved 100% accuracy.

## Libraries
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- ucimlrepo

## Run Project

```bash
pip install -r requirements.txt
python main.py


Create `requirements.txt`:

```txt id="k3m8s1"
numpy
pandas
matplotlib
seaborn
scikit-learn
ucimlrepo




pwd 'C:\\Users\\91831'
# ========================================================= # MACHINE LEARNING PROJECT
# WINE DATASET
# =========================================================
# INSTALL LIBRARIES
# pip install scikit-learn matplotlib seaborn pandas numpy ucimlrepo
# ========================================================= # IMPORT LIBRARIES
# =========================================================
import warnings warnings.filterwarnings("ignore")
import numpy as np import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import ( StandardScaler,
LabelEncoder
)
from sklearn.metrics import *
# ========================================================= # REGRESSION MODELS
# =========================================================
from sklearn.linear_model import ( LinearRegression, LogisticRegression,
Ridge, Lasso, Perceptron
)
# ========================================================= # CLASSIFICATION MODELS
# =========================================================
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ( DecisionTreeClassifier, plot_tree
)
from sklearn.naive_bayes import GaussianNB from sklearn.svm import SVC
# ========================================================= # ENSEMBLE MODELS
# =========================================================
from sklearn.ensemble import ( RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
# ========================================================= # NEURAL NETWORK
# =========================================================
from sklearn.neural_network import MLPClassifier
# ========================================================= # PCA
# =========================================================
from sklearn.decomposition import PCA
# ========================================================= # CLUSTERING
# =========================================================
from sklearn.cluster import ( KMeans,
DBSCAN,
AgglomerativeClustering
)
# ========================================================= # LOAD DATASET
# WINE DATASET
# =========================================================
dataset = fetch_ucirepo(id=109)
 
X = dataset.data.features.copy()
y = dataset.data.targets.squeeze().copy()
# Encode Target
y = LabelEncoder().fit_transform(y) print("Dataset Shape:", X.shape) print("\nFirst 5 Rows:\n") print(X.head())
# ========================================================= # MISSING VALUES
# =========================================================
print("\nMissing Values:\n") print(X.isnull().sum())
# ========================================================= # DATASET INFO
# =========================================================
print("\nDataset Information:\n") print(X.info())
# ========================================================= # HEATMAP
# =========================================================
plt.figure(figsize=(12,8))
sns.heatmap(
X.corr(), annot=True, cmap="coolwarm"
)
plt.title("Correlation Heatmap") plt.show()
# ========================================================= # OUTLIER DETECTION
# =========================================================
plt.figure(figsize=(14,6))
 

sns.boxplot(data=X) plt.xticks(rotation=90) plt.title("Outlier Detection") plt.show()
# ========================================================= # DISTRIBUTION PLOTS
# =========================================================
X.hist(
figsize=(14,10), bins=20
)
plt.suptitle("Feature Distribution") plt.show()
# ========================================================= # FEATURE SCALING
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ========================================================= # PCA
# =========================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled) plt.figure(figsize=(8,6))
plt.scatter(
X_pca[:,0],
X_pca[:,1], c=y
)
plt.title("PCA Visualization") plt.xlabel("PC1") plt.ylabel("PC2")
 

plt.grid() plt.show()
# ========================================================= # TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split( X_scaled,
y, test_size=0.2, random_state=42, stratify=y
)
# ========================================================= # ALL MODELS
# =========================================================
models = {
"Linear Regression": LinearRegression(),
"Logistic Regression": LogisticRegression(max_iter=5000),
"Ridge Regression": Ridge(alpha=1),
"Lasso Regression": Lasso(alpha=0.01),
"KNN":
KNeighborsClassifier(n_neighbors=5),
"Decision Tree": DecisionTreeClassifier(
max_depth=5, random_state=42
),
"Naive Bayes": GaussianNB(),
"SVM Linear": SVC(kernel="linear"),
"SVM RBF":
 
SVC(kernel="rbf"),
"Random Forest": RandomForestClassifier(
n_estimators=200, random_state=42
),
"AdaBoost": AdaBoostClassifier(
n_estimators=100, random_state=42
),
"Gradient Boosting": GradientBoostingClassifier(),
"Extra Trees": ExtraTreesClassifier(
n_estimators=200, random_state=42
),
"Neural Network": MLPClassifier(
hidden_layer_sizes=(64,32), max_iter=2000, random_state=42
),
"Perceptron": Perceptron()
}
# ========================================================= # STORAGE
# =========================================================
model_names = [] model_scores = []
# ========================================================= # TRAIN MODELS
# =========================================================
for name, model in models.items(): print("\n================================================")
 
print(name) print("================================================")
# Train
model.fit(X_train, y_train)
# Prediction
pred = model.predict(X_test)
# Regression Adjustment
if name in [
"Linear Regression", "Ridge Regression", "Lasso Regression"
]:
pred = np.round(pred).astype(int)
pred = np.clip( pred, y.min(),
y.max()
)
# Accuracy
accuracy = accuracy_score( y_test,
pred
)
model_names.append(name) model_scores.append(accuracy) print("\nAccuracy:", accuracy) # Classification Report
print("\nClassification Report:\n")
print(
classification_report( y_test,
pred
 
)
)
# ===================================================== # CONFUSION MATRIX
# =====================================================
cm = confusion_matrix( y_test,
pred
)
plt.figure(figsize=(6,4))
sns.heatmap( cm, annot=True, fmt="d",
cmap="Blues"
)
plt.title(f"{name} Confusion Matrix") plt.xlabel("Predicted") plt.ylabel("Actual")
plt.show()
# ===================================================== # DECISION TREE VISUALIZATION
# =====================================================
if name == "Decision Tree": plt.figure(figsize=(18,8))
plot_tree(
model, filled=True, fontsize=8
)
plt.title("Decision Tree") plt.show()
# ===================================================== # FEATURE IMPORTANCE
# =====================================================
 
if name in [
"Random Forest", "Extra Trees", "Gradient Boosting"
]:
importance = model.feature_importances_
indices = np.argsort( importance
)[::-1]
plt.figure(figsize=(10,5))
plt.bar(
range(X.shape[1]), importance[indices]
)
plt.xticks(
range(X.shape[1]), X.columns[indices], rotation=90
)
plt.title(
f"{name} Feature Importance"
)
plt.show()
# ===================================================== # NEURAL NETWORK LOSS
# =====================================================
if name == "Neural Network": plt.figure(figsize=(6,4))
plt.plot(
model.loss_curve_
)
plt.title(
"Neural Network Loss Curve"
)
plt.xlabel("Iterations")
 
plt.ylabel("Loss") plt.grid() plt.show()
# ========================================================= # K-MEANS
# =========================================================
kmeans = KMeans( n_clusters=3, random_state=42
)
kmeans_labels = kmeans.fit_predict(X_scaled) plt.figure(figsize=(8,6))
plt.scatter(
X_pca[:,0],
X_pca[:,1], c=kmeans_labels
)
plt.title("K-Means Clustering") plt.grid()
plt.show()
# ========================================================= # ELBOW METHOD
# =========================================================
wcss = []
for i in range(1,11):
km = KMeans(
n_clusters=i, random_state=42
)
km.fit(X_scaled) wcss.append(km.inertia_)
plt.figure(figsize=(8,5)) plt.plot(
 
range(1,11), wcss, marker='o'
)
plt.title("Elbow Method") plt.xlabel("Clusters") plt.ylabel("WCSS") plt.grid()
plt.show()
# ========================================================= # DBSCAN
# =========================================================
dbscan = DBSCAN( eps=2, min_samples=5
)
dbscan_labels = dbscan.fit_predict(X_scaled) plt.figure(figsize=(8,6))
plt.scatter(
X_pca[:,0],
X_pca[:,1], c=dbscan_labels
)
plt.title("DBSCAN Clustering") plt.grid()
plt.show()
# ========================================================= # HIERARCHICAL CLUSTERING
# =========================================================
hc = AgglomerativeClustering( n_clusters=3
)
hc_labels = hc.fit_predict(X_scaled) plt.figure(figsize=(8,6))
 

plt.scatter(
X_pca[:,0],
X_pca[:,1], c=hc_labels
)
plt.title("Hierarchical Clustering") plt.grid()
plt.show()
# ========================================================= # MODEL COMPARISON BAR GRAPH
# =========================================================
plt.figure(figsize=(16,6))
bars = plt.bar( model_names, model_scores
)
plt.title("Model Accuracy Comparison") plt.ylabel("Accuracy") plt.xticks(rotation=45)
for bar in bars:
yval = bar.get_height()
plt.text(
bar.get_x() + 0.1,
yval + 0.01,
round(yval, 2)
)
plt.grid(axis="y") plt.show()
# ========================================================= # MODEL COMPARISON LINE GRAPH
# =========================================================
plt.figure(figsize=(16,6)) plt.plot(
 
model_names, model_scores, marker='o'
)
plt.title("Model Accuracy Line Plot") plt.ylabel("Accuracy") plt.xticks(rotation=45)
plt.grid() plt.show()
# ========================================================= # PIE CHART
# =========================================================
plt.figure(figsize=(10,10))
plt.pie(
model_scores, labels=model_names, autopct='%1.1f%%'
)
plt.title("Model Accuracy Share") plt.show()
# ========================================================= # CLASS DISTRIBUTION
# =========================================================
plt.figure(figsize=(6,4)) sns.countplot(x=y) plt.title("Class Distribution") plt.show()
# ========================================================= # BEST MODEL
# =========================================================
best_index = np.argmax(model_scores) print("\n================================================")
 
print("BEST MODEL:", model_names[best_index]) print("BEST ACCURACY:", model_scores[best_index])
print("================================================")
# ========================================================= # END
# =========================================================
Dataset Shape: (178, 13)
First 5 Rows:

Alcohol Malicacid	Ash Alcalinity_of_ash Magnesium Total_phenols \
0	14.23	1.71	2.43	15.6	127
2.80					
1	13.20	1.78	2.14	11.2	100
2.65					
2	13.16	2.36	2.67	18.6	101
2.80					
3	14.37	1.95	2.50	16.8	113
3.85					
4	13.24	2.59	2.87	21.0	118
2.80					

Flavanoids Nonflavanoid_phenols Proanthocyanins Color_intensity Hue \
0	3.06	0.28	2.29	5.64
1.04				
1	2.76	0.26	1.28	4.38
1.05				
2	3.24	0.30	2.81	5.68
1.03				
3	3.49	0.24	2.18	7.80
0.86				
4	2.69	0.39	1.82	4.32
1.04				

0D280_0D315_of_diluted_wines
0	3.92		Proline
1065	
1	3.40		1050	
2	3.17		1185	
3	3.45		1480	
4	2.93		735	

Missing Values:				
Alcohol		0		
 





















#	Column
Non-Null Count
Dtype	
				
0	Alcohol	178 non-null	float64
1	Malicacid	178 non-null	float64
2	Ash	178 non-null	float64
3	Alcalinity_of_ash	178 non-null	float64
4	Magnesium	178 non-null	int64	
5	Total_phenols	178 non-null	float64
6	Flavanoids	178 non-null	float64
7	Nonflavanoid_phenols	178 non-null	float64
8	Proanthocyanins	178 non-null	float64
9	Color_intensity	178 non-null	float64
10	Hue	178 non-null	float64
11	0D280_0D315_of_diluted_wines	178 non-null	float64
12	Proline	178 non-null	int64	
 
 
 
 
 
 
 
 













0	0.80	1.00	0.89	12
1	0.92	0.79	0.85	14
2	1.00	0.90	0.95	10
				
accuracy			0.89	36
macro avg	0.91	0.90	0.89	36
weighted avg	0.90	0.89	0.89	36
 
 













0	1.00	1.00	1.00	12
1	0.93	1.00	0.97	14
2	1.00	0.90	0.95	10
				
accuracy			0.97	36
macro avg	0.98	0.97	0.97	36
weighted avg	0.97	0.97	0.97	36
 
 













0	0.80	1.00	0.89	12
1	0.92	0.79	0.85	14
2	1.00	0.90	0.95	10
				
accuracy			0.89	36
macro avg	0.91	0.90	0.89	36
weighted avg	0.90	0.89	0.89	36
 
 













0	0.80	1.00	0.89	12
1	0.92	0.79	0.85	14
2	1.00	0.90	0.95	10
				
accuracy			0.89	36
macro avg	0.91	0.90	0.89	36
weighted avg	0.90	0.89	0.89	36
 
 













0	1.00	1.00	1.00	12
1	1.00	0.93	0.96	14
2	0.91	1.00	0.95	10
				
accuracy			0.97	36
macro avg	0.97	0.98	0.97	36
weighted avg	0.97	0.97	0.97	36
 
 













0	1.00	0.92	0.96	12
1	0.88	1.00	0.93	14
2	1.00	0.90	0.95	10
				
accuracy			0.94	36
macro avg	0.96	0.94	0.95	36
weighted avg	0.95	0.94	0.94	36
 
 


 
					
		precision	recall	f1-score	support
					
	0	0.92	1.00	0.96	12
	1	1.00	0.93	0.96	14
	2	1.00	1.00	1.00	10
					
	accuracy			0.97	36
	macro avg	0.97	0.98	0.97	36
	weighted avg	0.97	0.97	0.97	36















0	0.92	1.00	0.96	12
1	0.93	0.93	0.93	14
2	1.00	0.90	0.95	10
 
					
	accuracy			0.94	36
	macro avg	0.95	0.94	0.95	36
	weighted avg	0.95	0.94	0.94	36















0	1.00	1.00	1.00	12
1	0.93	1.00	0.97	14
2	1.00	0.90	0.95	10
				
accuracy			0.97	36
macro avg	0.98	0.97	0.97	36
weighted avg	0.97	0.97	0.97	36
 
 













0	1.00	1.00	1.00	12
1	1.00	1.00	1.00	14
2	1.00	1.00	1.00	10
				
accuracy			1.00	36
macro avg	1.00	1.00	1.00	36
weighted avg	1.00	1.00	1.00	36
 
 
 
 













0	0.86	1.00	0.92	12
1	0.92	0.86	0.89	14
2	1.00	0.90	0.95	10
				
accuracy			0.92	36
macro avg	0.93	0.92	0.92	36
weighted avg	0.92	0.92	0.92	36
 
 













0	0.92	1.00	0.96	12
1	0.93	0.93	0.93	14
2	1.00	0.90	0.95	10
				
accuracy			0.94	36
macro avg	0.95	0.94	0.95	36
weighted avg	0.95	0.94	0.94	36
 
 
 
 













0	1.00	1.00	1.00	12
1	1.00	1.00	1.00	14
2	1.00	1.00	1.00	10
				
accuracy			1.00	36
macro avg	1.00	1.00	1.00	36
weighted avg	1.00	1.00	1.00	36
 
 
 
 













0	1.00	1.00	1.00	12
1	0.93	1.00	0.97	14
2	1.00	0.90	0.95	10
				
accuracy			0.97	36
macro avg	0.98	0.97	0.97	36
weighted avg	0.97	0.97	0.97	36
 
 


 









0	1.00	0.92	0.96	12
1	0.88	1.00	0.93	14
2	1.00	0.90	0.95	10
				
accuracy			0.94	36
macro avg	0.96	0.94	0.95	36
weighted avg	0.95	0.94	0.94	36


 
 
 
 
 
 
 
 

 
 
 
 
 
 


