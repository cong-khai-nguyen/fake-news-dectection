# Fake News Dectection

* train.csv: A full training dataset with the following attributes: 
> > * id: unique id for a news article
> > 
> > * author: author of the news article
> > 
> > * text: the text of the article; could be incomplete
> > 
> > * label: a label that marks the article as potentially unreliable
> > > * 1: unreliable
> > > * 0: reliable

# Description
In this project, I use scikit-learn's multinomial Naive Bayes classifer to classify news articles into fake or real categories from the text of each given article. The reason I use multinomial Naive Bayes is because it is well-known and suitable for classification with discrete features like word counts for text classification. One thing that I learn from this project is the use of stop words and multinomial Naive Bayes classifer. Without any parameter tuning, I am able to achieve 92% accuracy on the model. I even try out the passive agressive classifier since there are a lot of great write-ups about how linear model work well with count vectorizer.

Resources needed:

[Learn the basic of CountVectorizer](https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c)

[CountVectorizer API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

[Multinomial Naive Bayes Explained](https://www.upgrad.com/blog/multinomial-naive-bayes-explained/)

[Passive Aggressive Classifer](https://thecleverprogrammer.com/2021/02/10/passive-aggressive-classifier-in-machine-learning/)

[Tutorial](https://www.datacamp.com/community/tutorials/scikit-learn-fake-news)

[Data](https://www.kaggle.com/competitions/fake-news/data)

