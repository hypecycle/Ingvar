import os
import sys
import nltk.classify.util
#from nltk.classify import NaiveBayesClassifier
from nltk import word_tokenize

import pickle
from operator import itemgetter 
import string #for losing punctuation

# import libraries in lib directory
base_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(base_path, 'lib'))

from flask import Flask, render_template, flash, request, session
#werkzeug.debug works, disable in production
from werkzeug.debug import DebuggedApplication
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

app = Flask(__name__)
app.debug = True
app.secret_key = 'DFBK2L9X6nD,sqfFkCaJtN'
app.wsgi_app = DebuggedApplication(app.wsgi_app, True)

class fragIngvar(Form):
    ingvarSatz = TextField('Dein Ingvar-Satz:', validators=[validators.required(message='Darf nicht leer sein'), validators.Length(min=10, max=1000, message='Muss zwischen 10 und 1000 Zeichen enthalten.')])

#loading pre trained classifier
with open('SentimentAnalysisClassifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
    f.close()

def word_feats(words):
    return dict([(word.lower(), True) for word in words])

@app.route('/', methods=['GET', 'POST'])
def home():
	testSentenceRaw = ''

	global classifier

	if request.method == 'POST':
		testSentenceRaw = request.form['ingvarSatz']
	else:
		testSentenceRaw = 'Die schärfsten Kritiker der Elche waren früher selber welche.'

	form = fragIngvar(request.form)
	session.pop('_flashes', None)

	'''if not ingvarSatz:
		testSentenceRaw = 'Die schärfsten Kritiker der Elche waren früher selber welche.'
	else:
		testSentenceRaw = ingvarSatz'''

	predIkea = 0
	translator = str.maketrans('', '', string.punctuation) #strip punctuation


	summe = [0,0]

	testSentence = testSentenceRaw.translate(translator) #strip punctuation
	testSentence_tokensRaw = word_tokenize(testSentence)
	testSentence_tokens = list(set(testSentence_tokensRaw)) #removes duplicate words

	wortListe = []

	for test_word in testSentence_tokens:
		wortListeItem = [] #construct line
		wortListeItem.append(test_word)
		j = 1 #helper for list summe
		dist = classifier.prob_classify(word_feats([test_word]))
		
		for i in list(dist.samples()):
			summe[j] += dist.prob(i) #iterating through sets
			wortListeItem.append(dist.prob(i))
			j -= 1
		wortListe.append(wortListeItem)	#contsruct line

	#print(summe)
	#print(wortListe)

	#building list according to style
	if summe[0] > summe[1]:
		wortListe.sort(key = itemgetter(1), reverse = False)
	else:
		wortListe.sort(key = itemgetter(2), reverse = False)

	summeIkea = int(summe[1] / (summe[0] + summe[1]) * 100)


	return render_template('ingvar.html',
    						testSentence = testSentenceRaw,
    						predIkea = summeIkea,
    						pieData = summe,
    						wortListe = wortListe,
    						form=form,
                            )
