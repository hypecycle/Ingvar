import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import PlaintextCorpusReader, ConllCorpusReader
from nltk import word_tokenize

import pickle



def word_feats(words):
    return dict([(word.lower(), True) for word in words])


###
#Abschnitt Naive Bayes tranieren
###

print("Lese neutralen Korpus")
neutralCorp = ConllCorpusReader('.', 'corpora/tiger_release_aug07.corrected.16012013.conll09',['ignore', 'words', 'ignore', 'ignore', 'pos'],encoding='utf-8')

print("Lese Ingvar-Korpus")
ingvarCorp = PlaintextCorpusReader(".", "texts/latest.txt")

print("Generiere Wortlisten")
ingvarSentencesLong = ingvarCorp.sents()
neutralSentencesLong = neutralCorp.sents()

smallerSet = min(len(ingvarSentencesLong), len(neutralSentencesLong))

ingvarSentences = ingvarSentencesLong[:smallerSet]
neutralSentences = neutralSentencesLong[:smallerSet]

print(f'Zahl der Sätze limitiert auf kleineres Set mit {smallerSet} Sätzen')

print("Generiere Features")
ingFeats = [(word_feats(f), 'ing') for f in ingvarSentences]
neutFeats = [(word_feats(f), 'neu') for f in neutralSentences]
 
print("Generiere Cutoff")
ingCutoff = int(len(ingFeats)*0.9)
neutCutoff = int(len(neutFeats)*0.9)

print(f'Sätze Ingvar-Korpus {len(ingFeats)}, Sätze neutraler Korpus {len(neutFeats)}')

print("Trainiere Classifier mit Kontrollmenge")
trainfeats = ingFeats[:ingCutoff] + neutFeats[:neutCutoff]
testfeats = ingFeats[ingCutoff:] + neutFeats[neutCutoff:]
print('Trainiere mit %d Features, Teste mit %d Features' % (len(trainfeats), len(testfeats)))
 
classifierTrain = NaiveBayesClassifier.train(trainfeats)
print('Genauigkeit:', nltk.classify.util.accuracy(classifierTrain, testfeats))
classifierTrain.show_most_informative_features()

print("Trainiere Classifier zum Weiterverwenden")
mainFeats = ingFeats + neutFeats
classifier = NaiveBayesClassifier.train(mainFeats)


with open('SentimentAnalysisClassifier.pickle', 'wb') as f:
	pickle.dump(classifier, f, protocol = 2)
	f.close()


"""
test_sentence = ''

while test_sentence.lower() is not 'stop':

	summe = {'neu': 0, 'ing': 0}

	test_sentence = input('Dein Ingvar-Satz > ')

	test_sentence_tokens = word_tokenize(test_sentence)

	for test_word in test_sentence_tokens:
		#print(test_word)
		dist = classifier.prob_classify(word_feats([test_word]))
		for i in list(dist.samples()):
			#print(i, dist.prob(i))
			summe[i] += dist.prob(i)

	print(summe)"""

pass





