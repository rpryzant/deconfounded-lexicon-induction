# Causal selection 

A Python module that implements the Directed Residualization algorithms from 
1. _Deconfounded Lexicon Induction for Interpretable Social Science_ [(Pryzant, et al. 2019)](https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf)
2. _Interpretable Neural Architectures for Attributing an Ad’s Performance to its Writing Style_ [(Pryzant, et al. 2019)](https://nlp.stanford.edu/pubs/pryzant2018emnlp.pdf)

Given some text, ngram features, outcomes, and confounds, the algorithm scores each ngram according to how well it explains the outcomes _while controlling for confounds_. 


## Install

```
$ pip install causal_selection
```

## Test

You can use `test.py` and the included data to run an integration test.

```
$ python3 test.py
```
No output means the test passed.

## Use
The module exposes two functions: `score_vocab` and `evaluate_vocab`.

#### `score_vocab(text, vocab, confound_data, outcome_data, confound_names, outcome_names)`

**Arguments**
* **text**: list(list(str)). Input text that's **already been tokenized**
* **vocab**: list(string). The vocabulary to score. For **ngrams**, you can combine words with a space, e.g. `['a', 'b', 'a b']`.
* **confound_data**: list(list(float) --or-- list(string) ). Data for one or more confounds. These data can be categorical 
            (e.g. `['a', 'b', 'a']`) or continuous (e.g. `[1.0, 0.9, 0.1]`).
* **outcome_data**: list(list(float) --or-- list(string) ). Data for one or more outcomes. These data can be categorical 
            or continuous just like the confound data.
* **confound_names**: list(string). An optional list of names for each of the confound variables.
* **outcome_names**: list(string). An optional list of names for each of the outcome variables.

**Returns**
A mapping: outcome variable name => outcome variable class => a list of tuples containing vocab elements and their score (i.e. how important each feature is for that level of the outcome).

**Example**:
```
scores = score_vocab(
  text=[
    ["this", "is", "test" "1"],
    ["this", "is", "test", "2"]],
  vocab=["this", "is", "this is"],
  confound_data=[
    ["a", "b"],
    [0.1, 0.6]],
  outcome_data=[
    ["A", "B"],
    [0.5, 0.8]],
  confound_names=["C1", "C2"],
  outcome_names=["O1", "O2"]
)

# Now scores will look something like the following:
scores = {
  "O1": {
    "A": [
      ("this", -0.1),
      ("is", 0.0),
      ("this is", 1.0)
    ], ...
  }
}
```


#### `evaluate_vocab(text, vocab, confound_data, outcome_data)`

**Arguments**
These arguments are all the same as `score_vocab()`. 

**Returns**
The _informativeness coefficient_ of the vocab, which measures the strength of the text's causal effects that can be attributed to the vocab. 

## Tips

* For a continuous variable X, give the algorithm _log(X)_ instead of just X.
* The algorithm is sensitive to hyperparameter settings (number of training steps, hidden dimension, etc). Try several different settings to get the best scores possible.
