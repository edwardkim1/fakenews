import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

documents = [
    'this is the first document',
    'this is the second document',
    'this is some fake news',
    'this news is even faker than the last'
]

y = [0, 0, 1, 1]

vectorizer = CountVectorizer(stop_words='english')
word_counts = vectorizer.fit_transform(documents)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(word_counts)

clf = LogisticRegression(penalty='l1', C=1e5)
clf.fit(tfidf, y)

print(pd.Series(clf.coef_[0], index=vectorizer.get_feature_names())) # word coefficients

new_articles = ['This article is fake', 'Not sure about this one']
new_counts = vectorizer.transform(new_articles)
new_tfidf = transformer.transform(new_counts)
clf.predict_proba(new_tfidf)