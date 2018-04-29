import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import newspaper
from newspaper import Article
import nltk


import re
from flask import Flask, abort, redirect, render_template, request
from html import escape
from werkzeug.exceptions import default_exceptions, HTTPException


# Web app
app = Flask(__name__)


@app.after_request
def after_request(response):
    """Disable caching"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


def scrapeArticle(url):
    article = Article(url)

    article.download()
    article.parse()

    data = {}
    data['title'] = article.title
    data['text'] = article.text

    return data

def machineLearning(article):
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

    new_article = [scrapeArticle(article)["text"]]
    new_counts = vectorizer.transform(new_article)
    new_tfidf = transformer.transform(new_counts)

    return clf.predict_proba(new_tfidf)[0][0] * 100






@app.route("/")
def index():
    """Handle requests for / via GET (and POST)"""
    return render_template("index.html")


@app.route("/fakenews", methods=["POST"])
def fakenews():

    url = request.form.get("url")
    if not url:
        abort(400, "missing strings")

    value = machineLearning(url)

    return render_template("results.html", value = value)
