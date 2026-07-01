💳 Credit Card Fraud Detection

📌 Overview

This project detects fraudulent credit card transactions using Logistic Regression. Since the dataset is highly imbalanced, under sampling was applied to the training data after the train-test split.

🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

📊 Workflow

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Duplicate Removal
- Train-Test Split
- Under Sampling
- Feature Scaling
- Logistic Regression
- Model Evaluation
- Cross Validation

📈 Results

- Accuracy: ~97.9%
- Cross Validation Accuracy: ~93.5%
- Fraud Recall: ~87%
- Fraud Precision: ~7%

Note: The dataset is highly imbalanced. The model achieved high recall by detecting most fraudulent transactions, while precision remained relatively low due to a higher number of false positive predictions. This is a common trade-off in fraud detection problems.

📂 Dataset

The dataset is not included in this repository due to its large size.

Download Dataset:

https://drive.google.com/file/d/1YHmatsGkuZdcLqcC0sn5ynOt1DR9x1q4/view?usp=drivesdk

After downloading, place "creditcard.csv" in the project folder before running the notebook.

👩‍💻 Author

Khwahish 

⭐ If you found this project useful, consider giving it a star.
