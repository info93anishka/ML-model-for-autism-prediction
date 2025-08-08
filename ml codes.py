# train_all_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

'''from google.colab import files  comment out while using in colab'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

# Upload CSV manually
df = pd.read_csv("your_dataset.csv")  # Replace with your actual CSV filename
'''
use below commands for colab
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)  
''' 

# Load and clean dataset
df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})
df = df[df['result'] > -5]  # remove outliers

def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

# Age groups
df['ageGroup'] = df['age'].apply(convertAge)

# Feature engineering
def add_feature(data):
    data['sum_score'] = data.loc[:, 'A1_Score':'A10_Score'].sum(axis=1)
    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
    return data

df = add_feature(df)
df['age'] = df['age'].apply(lambda x: np.log(x))

# Label encode object columns
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

df = encode_labels(df)

# Prepare train-test split
removal = ['ID', 'age_desc', 'used_app_before', 'austim']
X = df.drop(removal + ['Class/ASD'], axis=1)
y = df['Class/ASD']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

# Balance dataset
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Normalize features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_val = scaler.transform(X_val)

# Models
models = [
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("SVM", SVC(kernel='rbf', probability=True)),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

# Train, evaluate and save
for name, model in models:
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(y_pred)

    print(f"\n{name}")
    print(f"Training AUC: {roc_auc_score(y_resampled, model.predict(X_resampled)):.4f}")
    print(f"Validation AUC: {roc_auc_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    # Save model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_val, y_proba):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of All Models")
plt.legend()
plt.show()
