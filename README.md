# fakenews
HackCrimson 2018 Kensho $1000 Grand Prize Winning Project

Edward S. Kim '21, Nazeli Hagen '21, William Yao '21 

This is a proof-of-concept flask web app that takes an article's url as input and outputs a percentage, which indicates
the likelihood that the article is "real." (i.e. a low percentage would mean the article is likely "fake")
The percentage is determined by a logistic regression model using scikit learn on a document-term matrix of 
10 washington post political news articles and 10 breitbart political news articles. This supervised learning approach 
assumes differences in language between fake news and normal news. For example, fake news articles may use more 
qualifying adjectives that are strong or lopsided in connotation.

Performance Analysis:

The app identifies onion.com articles and clickhole.com articles as fake, which is surprisingly good, considering
the training data was on breitbart political news and washington post political news only.
The app identifies most new york times political articles as not fake, which is also surprisingly consistent.
For new york times community news, such as the arrest of a robber, the app indicates a 50% likelihood of being fake.
This is interesting and is probably because new words are used to describe the arrest that are not commonly used in
political news; however, the words that are present most consistently in breitbart political news are not present in this
description of the arrest.

Problems to address:
1) What people define as "fake news" varies. If by "fake news" one means factually false statements, then
our model is unable to identify these false statements because there is no recognizable differences in word frequencies
between false and true statements. Also, how do we know any statement is true? We mostly believe in expert testimony
like a historian on facts about WWII. However, if the historian did not live in WWII, how do we believe his statements
about WWII to be true? He has collected evidence, but what is the threshold? Is a journal entry clear enough? What if
there are contradicting entries? What if the primary source author is known to fabricate stories, though his entries
seem reasonable? Then, is truth merely a consensus? Should computers simply look for the majority voice?
We realized that truth is still a difficult problem for a computer to determine because we also often
disagree on the definition of truth. Rather, for this project, we interpreted "fake news" to mean biased news, news sources 
that have a prejudiced lens through which the material is presented. The notion of bias also is tough to define as our perception
of the world in general is through a personal lens, and any writing will inherently require a lens. To determine which lenses
are "biased," we compared highly opinionated breitbart articles to relatively unopinionated washington post articles
of the same topics. Unfortunately, this could mean the ideal article is a static article, and any vibrant article would be 
deemed "fake." We need to test a larger dataset with articles from The New Yorker (for example), which are more lexically vibrant.
This introduces another problem: who is to be the determiner of biased and unbiased news in . In summary, the interesting question
seems to be how computers can fact check articles without any human assistance; however, due to difficulties in approaching this
problem, we indirectly aimed to do this by sorting biased and unbiased articles, which are reasonably correlated with "real" and
"fake" news.
2) The regression model currently uses a dataset of 10 washington post and 10 breitbart articles, which is
not the most representative set of the terms "unbiased" and "biased." These articles were handpicked.
3) Every query triggers a new instance of learning, which is inefficient because 
a) wait time is required for every query instance and
b) the web app is not scalable for larger data sets, which require a longer time for the computer to generate a model.
