import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('data/train.csv')

# print(df.head())

x = df.drop('label', axis = 1)
# print(x.head())
y = df['label']
# print(y.head())

print(df.shape)
# Check if there is null values indicating missing values
# print(df.isna())
# Drop rows where it has missing values
df.dropna(inplace=True)
print(df.shape)

copy = df.copy()
copy.reset_index(inplace=True)