from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
					])

import wikipedia
def get_data(keywords=['Facebook'], languages=['en', 'cz', 'de']):
	data_list = []
	for lang in languages:
		 wikipedia.set_lang(lang)
		 lang_sentence =""
		 for keyword in keywords:
		 	lang_sentence += wikipedia.summary(keyword)

		 data_list.append(lang_sentence)

	return (data_list, languages)




# set up data 
(data, target) = get_data(keywords=['google', 'Facebook', 'abba'])

#print(data)
#exit ()

text_clf.fit(data, target)

import pickle
pickle.dump(text_clf, open( "text_clf.pkl", "wb" ))

#print (text_clf.predict(['aller bei']))

