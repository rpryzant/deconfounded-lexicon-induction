# Causal attribution 

This package lets you attribute outcomes to text features while controlling for confounding factors.

For example, if you want to find the words in product descriptions that are most predictive of sales, features like brand names (e.g. "nike") will have the highest score. But these are words you can't change. This package will let you _control_ for confounds like brand and find which aspects of the description's _writing style_ are most predictive of sales. 

For more information see the blog post http://ai.stanford.edu/blog/text-causal-inference/

It has has two methods:

(1) **score_vocab**. Given text (T), vocab (V), outcome(s) Y, and confound(s) (C), this method will score each element of the vocab according to how well it explains each Y, controlling for all of the C's.

(2) **evaluate_vocab**. Measure's the strength of a vocab's causal effects on outcome Y (controlling for confounds C).

## Install

```
$ pip3 install causal-attribution
```

More info here: https://pypi.org/project/causal-attribution/

## Test

You can use `test.py` and the included data to run an integration test.

```
$ python3 test.py
```
No output means the test passed. This file also contains example usage.

## Examples

Let's say we have a file, `descriptions.csv`, which contains product descriptions for Nike and Addidas shoes:


| Description   | Brand   | Sales |
|---------------|---------|-------|
| buy shoes !     | addidas | 15    |
| fresh nike shoes !  | nike    | 35    |
| nice nike shoes ! | nike    | 17    |


We want to find the words that are most predictive of sales. Running a regression might give us `nike`, but this isn't super helpful, because brand names like "nike" are merely a function of confounding circumstance rather than a part of the writing style. We want the importance of each word while controlling for the influence of brand. The `score_vocab` function lets us do this:

```
import causal_attribution
importance_scores = causal_attribution.score_vocab(
	vocab=['buy', 'now' '!', 'nike', 'fresh', 'nice'],
	csv="descriptions.csv"
	name_to_type={
		'Description': 'input',
		'Brand': 'control',
		'Sales': 'predict',
	})
```

`importance_scores` will contain a list of `(word, score)` tuples.

If we want to evaluate the overal ability of our vocabulary's ability to make causal inferences about sales, we can use . `evaluate_vocab`:

```
import causal_attribution
informativeness = causal_attribution.evaluate_vocab(
	vocab=['buy', 'now' '!', 'nike', 'fresh', 'nice'],
	csv="descriptions.csv"
	name_to_type={
		'Description': 'input',
		'Brand': 'control',
		'Sales': 'predict',
	})
```
`informativeness` will be a float that reflects the vocabulary's abiltiy to predict sales, _beyond_ the brand's ability to predict sales.




## score_vocab

```
def score_vocab(
    vocab,
    csv="", delimiter="",
    df=None,
    name_to_type={},
    scoring_model="residualization",
    batch_size=128, train_steps=5000, lr=0.001,  hidden_size=32, max_seq_len=128,
    status_bar=False):
    """
    Score words in their ability to explain outcome(s), regaurdless of confound(s).

    Args:
        vocab: list(str). The vocabulary to use. Include n-grams
            by including space-serated multi-token elements
            in this list. For example, "hello world" would be a bigram.
        csv: str. Path to a csv of data. The column corresponding to 
            your "input" variable needs to be pre-tokenized text, where
            each token is separated by whitespace.
        delimiter: str. Delimiter to use when reading the csv.
        df: pandas.df. The data we want to iterate over. The columns of
            these data should be a superset of the keys in name_to_type.
        name_to_type: dict. A mapping from variable names to whether they are
            "input", "predict", or "control" variables.
            You can only have one "input" variable (the text).
            You can have 1+ "predict" and 1+ "control" variables,
                and they can be categorical or numerical datatypes.
        scoring_model: string. The type of model to score. One of
            ["residualization", "adversarial"]
        batch_size: int. Batch size for the scoring model.
        train_steps: int. How long to train the scoring model for.
        lr: float. Learning rate for the scoring model.
        hidden_size: int. Dimension of scoring model vectors.
        max_seq_len: int. Maximum length of text sequences.
        status_bar: bool. Whether to show status bars during model training.

    Returns:
        variable name => class name => [(feature name, score)] 
        Note that the lists are sorted in descending order.
	"score" means how important each feature is for that level of the outcome. 
    """
```

## evaluate_vocab

```

def evaluate_vocab(vocab,
        csv="", delimiter="",
        df=None,
        name_to_type={},
        max_seq_len=128):
    """Compute the informativeness coefficient for a vocabulary.
    This coefficient summarizes the vocab's ability to explain an outcome,
        regaurdless of confounders.

    Args:
        vocab: list(str). The vocabulary to use. Include n-grams
            by including space-serated multi-token elements
            in this list. For example, "hello world" would be a bigram.
        csv: str. Path to a csv of data. The column corresponding to 
            your "input" variable needs to be pre-tokenized text, where
            each token is separated by whitespace.
        delimiter: str. Delimiter to use when reading the csv.
        df: pandas.df. The data we want to iterate over. The columns of
            these data should be a superset of the keys in name_to_type.
        name_to_type: dict. A mapping from variable names to whether they are
            "input", "predict", or "control" variables.
            You can only have one "input" variable (the text).
            You can have 1+ "predict" and 1+ "control" variables,
                and they can be categorical or numerical datatypes.
        max_seq_len: int. Maximum length of text sequences.


    Returns:
        A float which may be used to evalutate the causal effects of the vocab. This is called
	the "informativeness coefficient" of the vocab in the paper. 
    """
 ```
Note that the arguments to `evaluate_vocab` are largely the same as `score_vocab`. 


### Tips

* For a continuous variable X, give the algorithm _log(X)_ instead of just X.
* The algorithm is sensitive to hyperparameter settings (number of training steps, hidden dimension, etc). Try several different settings to get the best scores possible.

### Citation

If you use this package, please include hte following citations:

This package is based on the following papers:

1. _Deconfounded Lexicon Induction for Interpretable Social Science_ [(Pryzant, et al. 2019)](https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf)
2. _Interpretable Neural Architectures for Attributing an Ad’s Performance to its Writing Style_ [(Pryzant, et al. 2019)](https://nlp.stanford.edu/pubs/pryzant2018emnlp.pdf)




