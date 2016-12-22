import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
print(twenty_train.target[:10])

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf_naive_bayes = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted_naive_bayes_1 = clf_naive_bayes.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted_naive_bayes_1):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


text_clf_naive_bayes = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()) ])
text_clf_naive_bayes = text_clf_naive_bayes.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=0)
docs_test = twenty_test.data

predicted_naive_bayes_2 = text_clf_naive_bayes.predict(docs_test)
print(np.mean(predicted_naive_bayes_2 == twenty_test.target))

text_clf_support_vector = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=0)), ])
_ = text_clf_support_vector.fit(twenty_train.data, twenty_train.target)
predicted_support_vector = text_clf_support_vector.predict(docs_test)
print(np.mean(predicted_support_vector == twenty_test.target))

print(metrics.classification_report(twenty_test.target, predicted_support_vector, target_names=twenty_test.target_names))

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3) }

gs_clf = GridSearchCV(text_clf_support_vector, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

