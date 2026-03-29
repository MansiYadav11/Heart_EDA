import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df=pd.read_csv('/content/Heart.csv')
df.head()

#EDA

df.columns
df.shape
df.info()
df.describe()

df.duplicated().sum()
df.isnull().sum()
df['AHD'].value_counts() #to check if it is equally distributed data
#sns.countplot(x=df['AHD'])
df['AHD'].value_counts().plot(kind='bar')

def plotting(var,num):
  plt.subplot(2,2,num)
  sns.histplot(df[var],kde=True)

plotting('Age',1) 
plotting('RestBP',2) 
plotting('Chol',3) 
plotting('MaxHR',4) 

plt.tight_layout()
