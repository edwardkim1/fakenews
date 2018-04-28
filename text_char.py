import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import newspaper
from newspaper import Article
import nltk

def scrapeArticle(url):
    article = Article(url)

    article.download()
    article.parse()

    data = {}
    data['title'] = article.title
    data['text'] = article.text

    return data

x_text_list = []
file = open("testset2.txt","r")
for line in file.readlines():
	x_text_list.append(scrapeArticle(line)["text"])

y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

vectorizer = CountVectorizer(stop_words='english')
word_counts = vectorizer.fit_transform(x_text_list)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(word_counts)

clf = LogisticRegression(penalty='l2', C=1e5)
clf.fit(tfidf, y)

print(pd.Series(clf.coef_[0], index=vectorizer.get_feature_names())) # word coefficients

test1 = "https://politics.theonion.com/trump-boys-beg-father-to-nominate-g-i-joe-action-figur-1825543511"
test2 = "https://www.washingtonpost.com/news/the-fix/wp/2018/04/28/trump-said-it-was-tough-to-watch-too-much-of-the-paralympics-was-it-derogatory/"
new_articles = [scrapeArticle(test1)["text"],scrapeArticle(test2)["text"]]
new_counts = vectorizer.transform(new_articles)
new_tfidf = transformer.transform(new_counts)
clf.predict_proba(new_tfidf)
print(clf.predict_proba(new_tfidf))
