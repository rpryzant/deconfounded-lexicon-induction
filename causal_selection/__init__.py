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
__version__ = 1.16

from collections import defaultdict, OrderedDict
from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function

import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

# Silence sklearn warnings.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VocabScoringModel(nn.Module):
    """
    Base class for scoring models.
    """
    def __init__(self, var_info, use_counts, hidden_size, use_gpu):
        """Initialize a DirectedResidualization model.

        Args:
            var_info: dict. A mapping between variable names and info about 
                that variable.
            use_counts: bool. Whether to use counts of features in sparse
                representations (false = binary indicators).
            hidden_size: int. Size of hidden vectors.
            use_gpu: bool. Whether to use gpu.
        """
        super(VocabScoringModel, self).__init__()

        self.use_gpu = use_gpu

        # Everything needs to be in order so that we know what variable 
        # each parameter corresponds too.
        self.ordered_names = sorted(list(var_info.keys()))

        self.hidden_size = hidden_size
        self.use_counts = use_counts

        self.input_info = next(
            (v for v in var_info.values() if v['type'] == 'input'))
        self.confound_info = {k: v for k, v in var_info.items() if v['control']}
        self.outcome_info = {
            k: v for k, v in var_info.items() if not v['control'] and v['type'] != 'input'
        }


    def build_predictors(self, input_size, output_info, hidden_size, num_layers):
        """ Builds multiple prediction heads based on output_info.

        Args:
            input_size: int. Size of input vectors.
            output_info: dict. {variable name => info about that variable}
            hidden_size: int. Size of hidden layers.
            num_layers: int. Number of hidden layers.

        Returns:
            A pair of nn.ModuleDicts that map variables to their predictions
                and losses, respectively.
        """
        def prediction_head(info):
            layers = []
            in_dim = input_size
            out_dim = len(info['vocab'])
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_size, bias=False))
                in_dim = hidden_size
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            return nn.Sequential(*layers)

        out_preds = {}
        out_losses = {}
        for name, info in output_info.items():
            out_preds[name] = prediction_head(info)
            if info['type'] == 'continuous':
                out_losses[name] = nn.MSELoss(reduction='mean') 
            else:
                out_losses[name] = nn.CrossEntropyLoss(reduction='mean')

        return nn.ModuleDict(out_preds), nn.ModuleDict(out_losses)

    def predict(self, input_vec, target_info, predictors, criterions, data):
        """ Use a vector to predict each of the outcomes.

        Args:
            input_vec: torch.FloatTensor [batch, features]. Inputs to the 
                prediction heads.
            target_into: dict. Information about the target variables.
            predictors: nn.ModuleDict. A mapping of variable names to their 
                prediction heads.
            criterions: nn.ModuleDict. A mapping of variable names to their criterions.
            data: dict. The raw inputs mapping variable name => data.

        Returns:
            Predictions on each of the targets and cumulative loss for all targets.
        """
        predictions = {}
        loss = 0
        for name, info in target_info.items():
            pred = predictors[name](input_vec)
            predictions[name] = pred
            if info['type'] == 'continuous':
                pred = torch.squeeze(pred, -1)
            loss += criterions[name](pred, data[name])
        return predictions, loss

    def interpret(self):
        """ Get importance scores for each feature and outcome. 

        Returns:
            variable name => class name => [(feature name, score)] 
            Note that the lists are sorted in descending order.
        """
        params = dict(
            (k, v.cpu().detach().numpy()) for k, v in self.named_parameters())
        out = defaultdict(lambda: defaultdict(list))

        # For each feature...
        for vi, feature in enumerate(self.input_info['vocab']):
            # For each level of each outcome...
            for outcome_name, outcome_info in self.outcome_info.items():
                for oi, outcome_level in enumerate(outcome_info['vocab']):
                    # Sum importance across hidden states
                    s = 0
                    for hi in range(self.hidden_size):
                        to_encoding =  params['W_in.weight'][hi, vi]
                        to_output = params[
                            'final_predictors.%s.0.weight' % outcome_name][oi, hi]
                        s += to_encoding * to_output

                    out[outcome_name][outcome_level].append((feature, s))

        for a in out:
            for b in out[a]:
                out[a][b].sort(key=lambda x: x[1], reverse=True)

        return out


class DirectedResidualization(VocabScoringModel):
    """
    This is an implementation of the Deep Residualization model from 
    "Interpretable Neural Architectures for Attributing an Ad's Performance 
        to its Writing Style"
        https://nlp.stanford.edu/pubs/pryzant2018emnlp.pdf

    and

    "Deconfounded Lexicon Induction for Interpretable Social Science"
        https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf

    The model takes text T, confound(s) C, and outcome(s) Y.
    It uses C to predict Y, then concatenates those predictions (Yhat) 
        to an encoding of the text to predict Y a second time (Yhat').
    We then trace paths through the parameters of the model to 
        determing text feature importance.
    """
    def __init__(self, var_info, use_counts, hidden_size, use_gpu):
        """Initialize a DirectedResidualization model.

        Args:
            var_info: dict. A mapping between variable names and info about 
                that variable.
            use_counts: bool. Whether to use counts of features in sparse
                representations (false = binary indicators).
            hidden_size: int. Size of hidden vectors.
        """
        super(DirectedResidualization, self).__init__(
            var_info, use_counts, hidden_size, use_gpu)

        # Text => T
        self.W_in = nn.Linear(
            len(self.input_info['vocab']), self.hidden_size, bias=False)

        # C => y_hat
        c_vec_size = sum([len(info['vocab']) for info in self.confound_info.values()])
        self.confound_predictors, self.confound_criterions = self.build_predictors(
            input_size=c_vec_size,
            output_info=self.outcome_info,
            hidden_size=self.hidden_size,
            num_layers=2)

        # [T, y_hat] => y_hat_final
        t_yhat_size = self.hidden_size + \
            sum([len(info['vocab']) for info in self.outcome_info.values()])
        self.final_predictors, self.final_criterions = self.build_predictors(
            input_size=t_yhat_size,
            output_info=self.outcome_info,
            hidden_size=self.hidden_size,
            num_layers=1)

    def forward(self, batch):
        """ Forward pass.

        Args:
            batch: dict. {variable name => tensor}
                        - text inputs: torch.LongTensor [batch, max sequence length]
                        - Categorical variables: torch.LongTensor [batch]
                        - Continuous variables: torch.FloatTensor [batch]
        Returns:
            - Initial predictions and loss from the confounds only.
            - Final predictions and loss from the confounds + text.
        """
        # text => T
        text_bow = make_bow_vector(
            batch[self.input_info['name']], 
            len(self.input_info['vocab']), 
            self.use_counts, self.use_gpu)
        text_encoded = self.W_in(text_bow)

        # C => y_hat
        confound_input = glue_dense_vectors([
            (tensor, self.confound_info[name])
            for name, tensor in batch.items() if name in self.confound_info],
            self.use_gpu)
        confound_preds, confound_loss = self.predict(
            input_vec=confound_input, 
            target_info=self.outcome_info,
            predictors=self.confound_predictors, 
            criterions=self.confound_criterions,
            data=batch)

        # [T, y_hat] => y_hat_final
        final_input = torch.cat(
            [text_encoded] + \
            [confound_preds[n] for n in self.ordered_names if n in self.outcome_info],
            axis=-1)
        final_preds, final_loss = self.predict(
            input_vec=final_input, 
            target_info=self.outcome_info,
            predictors=self.final_predictors, 
            criterions=self.final_criterions,
            data=batch)

        return confound_preds, confound_loss, final_preds, final_loss


class AdversarialSelector(VocabScoringModel):
    """
    This is an implementation of the Deep Residualization model from 
    "Interpretable Neural Architectures for Attributing an Ad's Performance 
        to its Writing Style"
        https://nlp.stanford.edu/pubs/pryzant2018emnlp.pdf

    and

    "Deconfounded Lexicon Induction for Interpretable Social Science"
        https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf

    The model takes text T, confound(s) C, and outcome(s) Y.
    It uses C to predict Y, then concatenates those predictions (Yhat) 
        to an encoding of the text to predict Y a second time (Yhat').
    We then trace paths through the parameters of the model to 
        determing text feature importance.
    """
    def __init__(self, var_info, use_counts, hidden_size, use_gpu):
        """Initialize a DirectedResidualization model.

        Args:
            var_info: dict. A mapping between variable names and info about 
                that variable.
            use_counts: bool. Whether to use counts of features in sparse
                representations (false = binary indicators).
            hidden_size: int. Size of hidden vectors.
        """
        super(AdversarialSelector, self).__init__(
            var_info, use_counts, hidden_size, use_gpu)

        # Text => e
        self.W_in = nn.Linear(
            len(self.input_info['vocab']), self.hidden_size, bias=False)

        # e => C
        self.gradrev = ReversalLayer()
        self.confound_predictors, self.confound_criterions = self.build_predictors(
            input_size=hidden_size,
            output_info=self.confound_info,
            hidden_size=self.hidden_size,
            num_layers=1)

        # e => y_hat
        self.final_predictors, self.final_criterions = self.build_predictors(
            input_size=hidden_size,
            output_info=self.outcome_info,
            hidden_size=self.hidden_size,
            num_layers=1)

    def forward(self, batch):
        """ Forward pass.

        Args:
            batch: dict. {variable name => tensor}
                        - text inputs: torch.LongTensor [batch, max sequence length]
                        - Categorical variables: torch.LongTensor [batch]
                        - Continuous variables: torch.FloatTensor [batch]
        Returns:
            - Initial predictions and loss from the confounds only.
            - Final predictions and loss from the confounds + text.
        """
        # text => e
        text_bow = make_bow_vector(
            batch[self.input_info['name']], 
            len(self.input_info['vocab']), 
            self.use_counts, self.use_gpu)
        text_encoded = self.W_in(text_bow)

        # e => C_hat
        text_gradrev = self.gradrev(text_encoded)
        confound_input = glue_dense_vectors([
            (tensor, self.confound_info[name])
            for name, tensor in batch.items() if name in self.confound_info], 
            self.use_gpu)
        confound_preds, confound_loss = self.predict(
            input_vec=text_gradrev,
            target_info=self.confound_info,
            predictors=self.confound_predictors,
            criterions=self.confound_criterions,
            data=batch)

        # e => y_hat
        final_preds, final_loss = self.predict(
            input_vec=text_encoded,
            target_info=self.outcome_info,
            predictors=self.final_predictors,
            criterions=self.final_criterions,
            data=batch)

        return confound_preds, confound_loss, final_preds, final_loss


# From https://github.com/janfreyberg/pytorch-revgrad
class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

class ReversalLayer(nn.Module):
    def __init__(self):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__()

    def forward(self, input_):
        return RevGrad.apply(input_)


def glue_dense_vectors(tensors_info, use_gpu=False):
    """ Glue together a bunch of (possibly categorical/dense) vectors.

    Args:
        tensors_info: [(tensor, information about the variable), ...]
        use_gpu: bool. Whether to use gpu.
    Returns:
        torch.FloatTensor [vatch, feature size] -- all of the vectorized
            variables concatted together.
    """
    out = []
    for tensor, info in tensors_info:
        if info['type'] == 'categorical':
            vec = make_bow_vector(
                torch.unsqueeze(tensor, -1), len(info['vocab']), use_gpu=use_gpu)
            out.append(vec)
        else:
            out.append(torch.unsqueeze(tensor, -1))

    return torch.cat(out, 1)


def make_bow_vector(ids, vocab_size, use_counts=False, use_gpu=False):
    """ Make a sparse BOW vector from a tensor of dense ids.

    Args:
        ids: torch.LongTensor [batch, features]. Dense tensor of ids.
        vocab_size: vocab size for this tensor.
        use_counts: if true, the outgoing BOW vector will contain
            feature counts. If false, will contain binary indicators.
        use_gpu: bool. Whether to use gpu
    Returns:
        The sparse bag-of-words representation of ids.
    """
    vec = torch.zeros(ids.shape[0], vocab_size)
    ones = torch.ones_like(ids, dtype=torch.float)
    if use_gpu:
        vec = vec.cuda()
        ones = ones.cuda()
        ids = ids.cuda()
    vec.scatter_add_(1, ids, ones)
    vec[:, 1] = 0.0  # zero out pad
    if not use_counts:
        vec = (vec != 0).float()
    return vec


def get_info(examples, vocab=None, max_seq_len=256):
    """Gathers info on and creats a featurized example generator for a list of raw examples.

    Args:
        examples: list(list, float, or string). Examples to create generator for.
        vocab: list(str). A vocabulary for discrete datatypes (e.g. text or categorical).
        max_seq_len: int. maximum sequence length for text examples.

    Returns:
        A dict of info about this variable as well as a generator over featurized examples.
    """
    assert isinstance(examples, list), 'examples must be list; got ' + str(type(examples))
    assert len(examples) > 0, 'Empty example list!'

    # Text
    if isinstance(examples[0], list):
        assert vocab is not None, 'ERROR: must provide a vocab.'
        example_type = 'input'
        vocab = ['UNK', 'PAD'] + vocab
        tok2id = {tok: i for i, tok in enumerate(vocab)}
        ngrams = max(len(x.split()) for x in vocab)
        unk_id = 0

        def featurizer(example):
            ids = []
            for n in range(1, ngrams + 1):
                toks = [' '.join(example[i: i + n]) for i in range(len(example) - n + 1)]
                ids += [tok2id.get(x, 0) for x in toks]
            ids = ids[:max_seq_len]

            padded_ids = ids + ([1] * (max_seq_len - len(ids)))  # pad idx = 1
            return padded_ids

    # Continuous
    elif isinstance(examples[0], float) or isinstance(examples[0], int):
        example_type = 'continuous'
        vocab = ['N/A']
        if isinstance(examples[0], int):
            featurizer = lambda ex: float(ex)
        else:
            featurizer = lambda ex: ex

    # Categorical
    elif isinstance(examples[0], str):
        example_type = 'categorical'
        if not vocab:
            vocab = ['UNK'] + sorted(list(set(examples)))
        tok2id = {tok: i for i, tok in enumerate(vocab)}
        featurizer = lambda ex: tok2id.get(ex, 0)  # 0 is the unk id.

    else: 
        print("ERROR: unrecognized example type: ", examples[0])
        quit()

    return featurizer, example_type, vocab


def get_iterator(vocab, df, name_to_type, batch_size=32, max_seq_len=256):
    """Builds a data iterator for text, confounds, and outcomes.
    Args:
        vocab: list(str). The vocabulary to use.
        df: pandas.df. The data we want to iterate over. The columns of
            these data should be a superset of the keys in name_to_type.
        name_to_type: dict. A mapping from variable names to whether they are
            "input", "predict", or "control" variables.
        batch_size: int. The batch size to use.
        max_seq_len: int. Maximum length of text sequences.

    Returns: 
        A generator which yields dictionaries where variable names are mapped
            to tensors of batched data.
    """ 

    def featurize(featurizer):
        return [featurizer(ex) for ex in examples]

    var_info = defaultdict(lambda: OrderedDict())
    featurized_data = defaultdict(list)
    for var_name, var_type in name_to_type.items():

        examples = list(df[var_name])

        if var_type == 'input':
            examples = [x.split() for x in examples]
            featurizer, _, vocab = get_info(examples, vocab, max_seq_len)
            var_info[var_name] = {
                'control': False, 'name': var_name, 
                'type': var_type, 'vocab': vocab
            }

        else:
            featurizer, varType, vocab = get_info(examples)
            var_info[var_name] = {
                'control': var_type == 'control', 
                'name': var_name, 'type': varType, 'vocab': vocab
            }

        featurized_data[var_name] = [featurizer(ex) for ex in examples]

    def to_tensor(var_name):
        dtype = torch.float
        if var_info[var_name]['type'] in {'categorical', 'input'}:
            dtype = torch.long
        return torch.tensor(featurized_data[var_name], dtype=dtype)

    feature_names = sorted(featurized_data.keys())
    data = TensorDataset(*[to_tensor(name) for name in feature_names])
    dataloader = DataLoader(
        dataset=data,
        sampler=RandomSampler(data),
        collate_fn=lambda batch: [torch.stack(x) for x in zip(*batch)],  # group by datatype.
        batch_size=batch_size)

    def iterator():
        for batch in dataloader:
            yield dict(zip(feature_names, batch))

    return iterator, var_info


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
        raise Exception('Must provide a csv or df.')        

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
            model = linear_model.Ridge()
            metric = 'neg_mean_squared_error'
        else:
            Y = make_bow_vector(torch.unsqueeze(Y, -1), len(y_info['vocab']))
            Y = Y.cpu().detach().numpy()
            model = OneVsRestClassifier(linear_model.LogisticRegression())
            metric = 'neg_log_loss'

        C_error = -cross_val_score(model, C, Y, scoring=metric, cv=5).mean()

        XC = np.concatenate((X, C), axis=-1)
        XC_error = -cross_val_score(model, XC, Y, scoring=metric, cv=5).mean()

        out[outcome] = C_error - XC_error

    return out
