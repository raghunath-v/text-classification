from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

# Each document need to be preprocessed with stopwords and punctations before
def wk(document1, document2):

	vectorizer = CountVectorizer()
	# make a list of unique words from both documents and index each documents with the appearing words and its frequency
	# 
	data_feature = vectorizer.fit_transform([document1, document2]) 
	#print(data_feature)
	data_feature = data_feature.toarray()
	# print(data_feature)
	# print(len(data_feature[0]))
	# make tfidf
	tfidf = TfidfTransformer().fit_transform(data_feature).toarray()	
	# print(tfidf)
	return np.dot(tfidf[0], tfidf[1])


# default n-gram is tri-gram, each document need remove stopwords and punctuation		
def ngk(document1, document2, n=7):
	#print(len(document1))
	#print(len(document2))
	n_gram_char = CountVectorizer(analyzer="char", ngram_range=(n, n)).fit_transform([document1, document2]).toarray()
	

	#print(n_gram_char[0][:50])
	#print(n_gram_char[1][:50])
	#print(len(n_gram_char[0]))
	#print(len(n_gram_char[1]))
	
	n_gram_char_normalized = TfidfTransformer().fit_transform(n_gram_char).toarray()
	return np.dot(n_gram_char_normalized[0], n_gram_char_normalized[1])

