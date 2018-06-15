# ingvar
Predicting the style fit of a German test sentence with the typical IKEA brand tone of voice. 
A NLTK classifier is trained with a neutral corpus and a corpus containing large chunks of IKEA scraped website copy to predict the stle fit
of one or the other. Training is done with trainer.py. Then the classifier is pickled.
The classifier is used by a flask app that exposes a website to enter and display the style fit an give explanations.
