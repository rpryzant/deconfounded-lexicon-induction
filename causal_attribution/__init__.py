"""
This package has two interfaces:

(1) score_vocab(): Given text (T), vocab (V), outcome(s) Y, and 
    confound(s) (C), this method will score each element of the
    vocab according to how well it explains each Y, controlling 
    for all of the C's.

(2) evaluate_vocab(): Measure's the strength of a vocab's causal
    effects on Y (controlling for C).

(c) Reid Pryzant 2019 https://cs.stanford.edu/~rpryzant/
May be used and distributed under the MIT license.
"""
# TODOs
#    - loss weighting
#    - scheduling
#    - layers changeable
# https://packaging.python.org/tutorials/packaging-projects/

__all__ = ['score_vocab', 'evaluate_vocab']
__version__ = 1.01

import pandas as pd
import numpy as np
import scipy

import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

from .data import *
from .models import *

# Silence sklearn warnings.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def score_vocab(
    vocab,
    csv="", delimiter="",
    df=None,
    name_to_type={},
    scoring_model="residualization",
    batch_size=128, train_steps=5000, lr=0.001,  hidden_size=32, max_seq_len=128,
    status_bar=False,
    use_gpu=False):
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
        use_gpu: bool. Whether to use a gpu for model training.

    Returns:
        variable name => class name => [(feature name, score)] 
        Note that the lists are sorted in descending order.
    """
    if csv:
        df = pd.read_csv(csv, delimiter=delimiter).dropna()
    elif df is not None:
        df = df.dropna()
    else:
        raise Exception('Must provide a csv or df.')        

    assert 'UNK' not in vocab, 'ERROR: UNK is not allowed as vocab element.'
    assert 'PAD' not in vocab, 'ERROR: PAD is not allowed as vocab element.'

    iterator_fn, var_info = get_iterator(
        vocab, df, name_to_type,
        batch_size=batch_size,
        max_seq_len=max_seq_len)

    if scoring_model == 'residualization':
        model_fn = DirectedResidualization
    elif scoring_model == 'adversarial':
        model_fn = AdversarialSelector
    else:
        raise Exception("Unrecognized scoring_model: ", scoring_model)

    model = model_fn(
        var_info=var_info,
        use_counts=False,
        hidden_size=hidden_size,
        use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    iterator = iterator_fn()
    stepper = tqdm(range(train_steps)) if status_bar else range(train_steps)
    for i in stepper:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iterator_fn()

        if use_gpu:
            batch = {k: v.cuda() for k, v in batch.items()}
            
        confound_preds, confound_loss, final_preds, final_loss = model(batch)
        loss = confound_loss + final_loss  # TODO(rpryzant) weighting?

        loss.backward()
        optimizer.step()
        model.zero_grad()

    features_scores = model.interpret()

    return features_scores


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
        A float which may be used to evalutate the causal effects of the vocab.
    """
    if csv:
        df = pd.read_csv(csv, delimiter=delimiter).dropna()
    elif df is not None:
        df = df.dropna()
    else:
        raise Exception('ERROR: must provide a csv or df.')        

    if 'control' not in set(name_to_type.values()):
        raise Exception("ERROR: must control for at least one variable.")

    assert 'UNK' not in vocab, 'ERROR: UNK is not allowed as vocab element.'
    assert 'PAD' not in vocab, 'ERROR: PAD is not allowed as vocab element.'

    iterator_fn, var_info = get_iterator(
        vocab, df, name_to_type,
        batch_size=len(df),
        max_seq_len=max_seq_len)

    data = next(iterator_fn())

    input_name = next((k for k, v in name_to_type.items() if v == 'input'))
    X = make_bow_vector(data[input_name], len(var_info[input_name]['vocab']))
    X = X.cpu().detach().numpy()

    C = glue_dense_vectors([
        (tensor, var_info[name]) 
        for name, tensor in data.items() if var_info[name]['control']])
    C = C.cpu().detach().numpy()

    out = {}
    outcome_names = [k for k, v in name_to_type.items() if v == 'predict']

    for outcome in outcome_names:
        y_info = var_info[outcome]
        Y = data[outcome]
        if y_info['type'] == 'continuous':
            Y = Y.cpu().detach().numpy()
            model = linear_model.Ridge(fit_intercept=False)
            metric = 'neg_mean_squared_error'
        else:
            Y = make_bow_vector(torch.unsqueeze(Y, -1), len(y_info['vocab']))
            Y = Y.cpu().detach().numpy()
            model = OneVsRestClassifier(linear_model.LogisticRegression(fit_intercept=False))
            metric = 'neg_log_loss'

        C_error = -cross_val_score(model, C, Y, scoring=metric, cv=5).mean()

        XC = np.concatenate((X, C), axis=-1)
        XC_error = -cross_val_score(model, XC, Y, scoring=metric, cv=5).mean()

        out[outcome] = C_error - XC_error

    return out
