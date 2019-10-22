import causal_selection as selection



import nltk
from collections import Counter

def build_vocab(text, n=1000):
	all_words = [w for s in text for w in s]
	c = Counter(all_words)
	return [w for w, _ in c.most_common(n)]




data_reader = open('testdata/courses.tsv')


text = []

c1_categorical = []
c2_categorical = []
c3_continuous = []
c4_continuous = []
y1_categorical = []
y2_categorical = []
y3_continuous = []
y4_continuous = []

for i, l in enumerate(data_reader):
	parts = l.strip().split('\t')
	[
		description,
		title,
		subject,
		course_number,
		course_level,
		num_reqs,
		repeatable,
		grading, 
		units_min,
		units_max,
		level,
		final,
		course_id,
		section_id,
		term,
		component,
		num_enrolled,
		max_enrolled,
		num_waitlist,
		max_waitlist,
		add_consent,
		drop_consent,
		start_time,
		end_time,
		location,
		days,
		instructors
	] = parts

	try:
		new_text = [nltk.word_tokenize(description.lower())]
		new_c1_categorical = [subject]
		new_c2_categorical = [final]
		new_c3_continuous = [float(start_time)]
		new_c4_continuous = [float(end_time)]
		new_y1_categorical = [days]
		new_y2_categorical = [term.split()[-1]]
		new_y3_continuous = [float(num_enrolled)]
		new_y4_continuous = [float(end_time)]
	except:
		continue

	text += new_text
	c1_categorical += new_c1_categorical
	c2_categorical += new_c2_categorical
	c3_continuous += new_c3_continuous
	c4_continuous += new_c4_continuous
	y1_categorical += new_y1_categorical
	y2_categorical += new_y2_categorical
	y3_continuous += new_y3_continuous
	y4_continuous += new_y4_continuous

	if i > 10000: break

vocab = build_vocab(text)

scores = selection.score_vocab(
	text=text,
	vocab=vocab[:10],
	confound_data=[c1_categorical, c2_categorical, c3_continuous, c4_continuous],
 	outcome_data=[y1_categorical, y2_categorical, y3_continuous, y4_continuous],
 	confound_names=['subject', 'final', 'start', 'end'],
 	outcome_names=['instructors', 'term', 'enrolled', 'end'],
 	batch_size=2,
 	train_steps=500)


full_vocab_score = selection.evaluate_vocab(
	text=text,
	vocab=vocab,
	confound_data=[c2_categorical, c3_continuous],
 	outcome_data=y1_categorical)

partial_vocab_score = selection.evaluate_vocab(
	text=text,
	vocab=vocab[-50:],
	confound_data=[c2_categorical, c3_continuous],
 	outcome_data=y1_categorical)

assert full_vocab_score > partial_vocab_score
