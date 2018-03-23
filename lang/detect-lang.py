#!/usr/bin/python3

import pickle
import sys
text_clf = pickle.load(open( "text_clf.pkl", "rb" ))

sentence = ""
for token in sys.argv[1:]:
	sentence += token + " " 

print ("For sentence: %s" % sentence)
print ("Language detected: %s" % text_clf.predict([sentence]))