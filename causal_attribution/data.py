"""Data pipelines."""

from collections import defaultdict, OrderedDict
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch

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

