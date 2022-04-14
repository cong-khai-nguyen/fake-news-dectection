import pandas as pd
import numpy as np
import re as regex
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import itertools
import matplotlib.pyplot as plt
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

count_vect = CountVectorizer(max_features = 5000, ngram_range=(1, 3))
x = count_vect.fit_transform(corpus).toarray()

# print(x.shape)
# count_df = pd.DataFrame(data=x,columns = count_vect.get_feature_names())
# print(count_df)

y = copy['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
try:
    with open("model.pickle", "rb") as f:
        classifer = pickle.load(f)
except:
    classifer = MultinomialNB()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10)
    classifer.fit(x_train, y_train)
    with open("model.pickle", "wb") as f:
        pickle.dump((classifer), f)

# Credit: https://www.datacamp.com/community/tutorials/scikit-learn-fake-news
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


prediction = classifer.predict(x_test)
score = metrics.accuracy_score(y_test, prediction)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

