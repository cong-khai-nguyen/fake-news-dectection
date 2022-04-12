import pandas as pd
import re as regex
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
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

print(copy['title'][2])

# Replace all the special characters in the title column with space to create a bag of words
corpus = []

for i in range(len(copy)):
    review = regex.sub('[^a-z-A-Z]', ' ', copy['title'][i])
    review = review.lower()
    review = review.split()
    # stem all the words in the title if those words are not the stopwords list
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
