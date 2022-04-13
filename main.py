import pandas as pd
import re as regex
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# import nltk
# nltk.download('stopwords')
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

corpus = []

for i in range(len(copy)):
    # Replace all the special characters in the title column with space to create a bag of words
    review = regex.sub('[^a-zA-Z]', ' ', copy['title'][i])
    review = review.lower()
    review = review.split()
    # stem all the words in the title if those words are not the stopwords list
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # print(review)
    # Convert it back from array to string
    review = ' '.join(review)
    # print(review)
    corpus.append(review)

count_vect = CountVectorizer(max_features=5000, ngram_range=(1, 3))
x = count_vect.fit_transform(corpus).toarray()


df = pd.DataFrame(data=x,columns = count_vect.get_feature_names())
print(df)
