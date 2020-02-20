"""Vocab scoring models: DirectedResidualization and AdversarialSelector."""

from collections import defaultdict

import torch.nn as nn
import torch.optim as optim

from .utils import *


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
