import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('/content/Heart.csv')

# -------------------- BASIC EDA --------------------
print(df.head())

print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())

# -------------------- DATA CLEANING --------------------

# Remove duplicates
print("Duplicate values:", df.duplicated().sum())
df = df.drop_duplicates()

# Handle missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing values (if any)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# -------------------- TARGET DISTRIBUTION --------------------
print(df['AHD'].value_counts())
df['AHD'].value_counts().plot(kind='bar')
plt.title("Target Distribution (AHD)")
plt.show()

# -------------------- NUMERICAL FEATURE ANALYSIS --------------------
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)
    plt.title(var)

plt.figure(figsize=(10,8))
plotting('Age', 1) 
plotting('RestBP', 2) 
plotting('Chol', 3) 
plotting('MaxHR', 4) 
plt.tight_layout()
plt.show()

# -------------------- CATEGORICAL ANALYSIS --------------------
sns.countplot(x=df['Sex'])
plt.title("Sex Distribution")
plt.show()

sns.countplot(x=df['ChestPain'], hue=df['AHD'])
plt.title("Chest Pain vs AHD")
plt.show()

# -------------------- FEATURE ENGINEERING --------------------

# Convert categorical variables into numerical

# Binary encoding
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['AHD'] = df['AHD'].map({'Yes': 1, 'No': 0})
df['FastingBS'] = df['FastingBS'].map({1: 1, 0: 0})

# One-hot encoding for multi-category columns
df = pd.get_dummies(df, columns=['ChestPain', 'Thal'], drop_first=True)

# -------------------- FEATURE SCALING --------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak']
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------- FINAL DATA CHECK --------------------
print("Final dataset shape:", df.shape)
print(df.head())
