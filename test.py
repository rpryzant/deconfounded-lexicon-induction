""" Tests and examples for causal_selection. """
import causal_selection as selection
import random

import nltk
from collections import Counter

def build_vocab(text, n=1000):
	all_words = [w for s in text for w in s]
	c = Counter(all_words)
	return [w for w, _ in c.most_common(n)]

print('Reading data...')
data_reader = open('testdata/cfpb.1k.tsv')
next(data_reader)
text = []
for i, l in enumerate(data_reader):
	parts = l.strip().split('\t')
	[complaint, _, _, _, _, _, _, _] = parts
	text += [nltk.word_tokenize(complaint.lower())]

# Use a variety of variables (categorical and continuous) 
#  to score a vocab.
print('Scoring vocab...')

vocab = build_vocab(text)

scores = selection.score_vocab(
	vocab=vocab,
	csv="testdata/cfpb.1k.tsv", delimiter='\t',
	name_to_type={
		'consumer-complaint': 'input',
		'issue-in-question': 'control',
		'state-of-origin': 'control',
		'dummy-continuous-1': 'control',
		'timely-response': 'predict',
		'dummy-continuous-2': 'predict'
	},
 	batch_size=2,
 	train_steps=500)


print('Evaluating vocab...')
# Now evaluate 2 vocabs, and ensure that the larger
#  vocab is more informative.
full_scores = selection.evaluate_vocab(
	vocab=vocab,
	csv="testdata/cfpb.1k.tsv", delimiter='\t',
	name_to_type={
		'consumer-complaint': 'input',
		'dummy-continuous-1': 'control',
		'timely-response': 'control',
		'product-in-question': 'predict',
	})
partial_scores = selection.evaluate_vocab(
	vocab=[],
	csv="testdata/cfpb.1k.tsv", delimiter='\t',
	name_to_type={
		'consumer-complaint': 'input',
		'dummy-continuous-1': 'control',
		'timely-response': 'control',
		'product-in-question': 'predict',
	})

assert full_scores['product-in-question'] > partial_scores['product-in-question']

# And just for good measure have a continuous outcome.
partial_scores = selection.evaluate_vocab(
	vocab=[],
	csv="testdata/cfpb.1k.tsv", delimiter='\t',
	name_to_type={
		'consumer-complaint': 'input',
		'dummy-continuous-1': 'control',
		'timely-response': 'control',
		'product-in-question': 'predict',
	})


print('Tests passed!')
