import causal_selection as selection
import random


import nltk
from collections import Counter

def build_vocab(text, n=1000):
	all_words = [w for s in text for w in s]
	c = Counter(all_words)
	return [w for w, _ in c.most_common(n)]


print('Reading data...')
data_reader = open('testdata/cfpb.tsv')
text = []
products = []
issues = []
states = []
responses = []
times = []
rand0s = []
rand1s = []
for i, l in enumerate(data_reader):
	parts = l.strip().split('\t')
	[
		complaint, product,	issue,
		state, response, timely,
		rand0, rand1
	] = parts

	text += [nltk.word_tokenize(complaint.lower())]
	products += [product]
	issues += [issue]
	states += [state]
	responses += [response]
	times += [timely]
	rand0s += [float(rand0)]
	rand1s += [float(rand1)]

	if i > 40000:
		break


# Use a variety of variables (categorical and continuous) 
#  to score a vocab.
print('Scoring vocab...')
vocab = build_vocab(text)
scores = selection.score_vocab(
	text=text,
	vocab=vocab[:10],
	confound_data=[issues, states, rand0s],
 	outcome_data=[responses, rand1s],
 	confound_names=['issues', 'states', 'rand0s'],
 	outcome_names=['responses', 'rand1s'],
 	batch_size=2,
 	train_steps=500)


print('Evaluating vocab...')
# Now evaluate 2 vocabs, and ensure that the larger
#  vocab is more informative.
full_vocab_score = selection.evaluate_vocab(
	text=text,
	vocab=vocab,
	confound_data=[issues, rand0s],
 	outcome_data=responses)

partial_vocab_score = selection.evaluate_vocab(
	text=text,
	vocab=vocab[-50:],
	confound_data=[issues, rand0s],
 	outcome_data=responses)

assert full_vocab_score > partial_vocab_score

# And just for good measure make sure vocab evaluation
#  doesn't crash on continuous outcomes.
selection.evaluate_vocab(
	text=text,
	vocab=vocab[-50:],
	confound_data=[issues, rand0s],
 	outcome_data=rand1s)

print('Tests passed!')
