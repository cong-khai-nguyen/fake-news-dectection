# Fake News Dectection
Link:

# Description
In this project, I use scikit-learn's multinomial Naive Bayes classifer to classify news articles into fake or real categories from the text of each given article. The reason I use multinomial Naive Bayes is because it is well-known and suitable for classification with discrete features like word counts for text classification. One thing that I learn from this project is the use of stop words and multinomial Naive Bayes classifer. Without any parameter tuning, I am able to achieve 92% accuracy on the model. I even try out the passive agressive classifier since there are a lot of great write-ups about how linear model work well with count vectorizer.

Resources needed:

[Learn the basic of CountVectorizer](https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c)

[CountVectorizer API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

[Multinomial Naive Bayes Explained](https://www.upgrad.com/blog/multinomial-naive-bayes-explained/)

[Passive Aggressive Classifer](https://thecleverprogrammer.com/2021/02/10/passive-aggressive-classifier-in-machine-learning/)

[Tutorial](https://www.datacamp.com/community/tutorials/scikit-learn-fake-news)

[Data](https://www.kaggle.com/competitions/fake-news/data)

# Install and Run the Project
This project requires to imported and installed libraries: pandas, numpy, pickle, itertools, matplotlib, and scikit-learn.
