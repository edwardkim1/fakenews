from newspaper import *
import nltk


x_paper = newspaper.build('http://cnn.com')
print(x_paper.articles)
y_paper = newspaper.build('http://www.breitbart.com/')



def scrapeArticle(url):
    article = Article(url)

    article.download()
    article.parse()

    data = {}
    data['title'] = article.title
    data['text'] = article.text

    return data

def context(raw_text, word):
    tokens = nltk.word_tokenize(raw_text)
    text = nltk.Text(tokens)

    return text.concordance(word)



text1=scrapeArticle("https://www.washingtonpost.com/world/asia_pacific/north-and-south-korea-agree-to-work-toward-common-goal-of-denuclearization/2018/04/27/7dcb03d6-4981-11e8-8082-105a446d19b8_story.html?noredirect=on&utm_term=.4792c18d0bd2")
context(text1["text"], "best")

text2 = nltk.Text(nltk.word_tokenize(text1["text"]))

text2.dispersion_plot(["Trump", "democracy", "freedom", "Korea", "hope"])