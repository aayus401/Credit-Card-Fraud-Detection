💳 Credit Card Fraud Detection using Machine Learning

A comparative study of four classical machine learning algorithms — Logistic Regression, Decision Tree, K-Nearest Neighbors, and Support Vector Machine — applied to detecting fraudulent credit card transactions on a highly imbalanced, real-world dataset.

📌 Problem Statement

Credit card fraud accounts for billions of dollars in losses every year. The goal of this project is to build and evaluate multiple ML classifiers that can flag fraudulent transactions accurately, while dealing with the core challenge of fraud-detection datasets: severe class imbalance (fraudulent transactions make up a tiny fraction of all transactions).

📊 Dataset
Source: Kaggle — Credit Card Fraud Detection
Size: 284,807 transactions made by European cardholders over two days in September 2013
Class distribution: 492 fraudulent transactions (0.172%) vs. 284,315 legitimate transactions — a highly imbalanced dataset
Features: 31 columns total
V1–V28: anonymized principal components from a PCA transformation (original features withheld for confidentiality)
Time: seconds elapsed since the first transaction in the dataset
Amount: transaction amount
Class: target label (0 = legitimate, 1 = fraud)
🗂️ Repository Structure
Credit-Card-Fraud-Detection/
├── Credit Card Fraud Detection Using Machine Learning/
│   └── Code/
│       ├── Credit Card Fraud Detection - Logistic Regression.ipynb
│       ├── Credit Card Fraud Detection - Decision Tree.ipynb
│       ├── Credit Card Fraud Detection - K-Nearest Neighbor.ipynb
│       └── Credit Card Fraud Detection - Support Vector Machines.ipynb
├── Presentation/
│   └── Report-Presentation.pptx
├── LICENSE
└── README.md
🧪 Methodology

Each notebook explores a different algorithm and a different strategy for handling class imbalance:

Model	Imbalance handling	Train/Test split	Key steps
Logistic Regression	Random undersampling of the majority class to match the 492 fraud cases (984 balanced rows)	80/20	Baseline linear classifier on the balanced subset
Decision Tree	None (trained on full imbalanced data)	70/30	criterion='entropy'; evaluated with precision/recall/F1, not just accuracy
K-Nearest Neighbors	None (trained on full imbalanced data)	70/30	Features scaled with StandardScaler; optimal k selected via an error-rate curve across k = 1–39
SVM	imblearn.RandomUnderSampler (0.5 sampling ratio)	80/20	Time dropped, Amount standardized; evaluated with ROC-AUC and precision-recall curves in addition to accuracy
📈 Results

⚠️ Note on accuracy: Because fraud is only 0.17% of the dataset, overall accuracy is a misleading metric on the full (imbalanced) data — a model that predicts "not fraud" for every transaction would still score ~99.8%. Precision, recall, and F1-score on the fraud class specifically are more meaningful indicators of real-world performance.

Model	Accuracy	Fraud-class Precision	Fraud-class Recall	Fraud-class F1
Logistic Regression (balanced subset)	91.9% (test)	—	—	—
Decision Tree (full imbalanced data)	~99.99%	0.83	0.81	0.82
K-Nearest Neighbors (full imbalanced data)	~99.9%+	see notebook	see notebook	see notebook
SVM (undersampled)	94.6–97.6%	0.99	0.86	0.92

The Decision Tree's confusion matrix on 85,443 test transactions: 128 fraud cases correctly caught, 30 missed, 26 legitimate transactions incorrectly flagged.

Across all four models, SVM and Decision Tree offered the best balance between catching fraud (recall) and minimizing false alarms (precision), while Logistic Regression on the undersampled data was a strong, interpretable baseline.

🛠️ Tech Stack
Language: Python
Libraries: pandas, NumPy, scikit-learn, imbalanced-learn (imblearn), seaborn, matplotlib
🚀 How to Run
Clone the repository:
bash
   git clone https://github.com/aayus401/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
Download creditcard.csv from the Kaggle dataset page and place it in the same folder as the notebooks.
Install dependencies:
bash
   pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib jupyter
Open any notebook in the Code/ folder and run all cells.
🔮 Future Work
Apply cost-sensitive learning or SMOTE-based oversampling as an alternative to undersampling, which discards a large amount of legitimate-transaction data
Incorporate ensemble methods (Random Forest, XGBoost) for comparison
Add cross-validation instead of a single train/test split for more robust metric estimates
Explore feature engineering using transaction location/velocity (e.g., flagging geographically implausible transaction sequences)
📜 License

MIT — feel free to fork, star, and use in your own portfolio.

👤 Author

Aayush (@aayus401)

## Conclusion
In conclusion, the main objective of this project was to find the most suited model for creditcard fraud detection in terms of the machine learning techniques chosen for the project. It was met by building the four models and finding the accuracies of them all; the best in terms of accuracy is KNN and Decision Tree, which scored 100 on credit card fraud and increased the customer’s satisfaction as it will provide themwith a better experience and feeling secure.

