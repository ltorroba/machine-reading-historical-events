from typing import List, Dict, Optional, Any, Tuple
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.attention import Attention

from allennlp.nn.util import get_text_field_mask, masked_softmax

from allennlp.training.metrics.mean_absolute_error import MeanAbsoluteError
from ..metrics import ExactMatchScore, KendallTauScore, PercentUnderYearRangeScore

from allennlp.modules.time_distributed import TimeDistributed

torch.manual_seed(1)


class ClassifierModule(nn.Module):
    def __init__(self, num_layers: int, hidden_features: int, in_features: int,
                 out_features:int, dropout: Optional[float] = None) -> None:
        super().__init__()
        self.dropout = dropout or 0.0
        self.classes = out_features

        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise Exception("Number of layers must be at least 1.")

        if num_layers <= 1:
            # Layer from in_features to 1, directly
            self.layers.extend([
                nn.Dropout(self.dropout),
                nn.Linear(in_features=in_features, out_features=out_features),
            ])
        else:
            # We add layers with hidden_features size
            self.layers.extend([
                nn.Dropout(self.dropout),
                nn.Linear(in_features=in_features, out_features=hidden_features),
                nn.PReLU()
            ])

            for i in range(num_layers - 2):
                self.layers.extend([
                    nn.Dropout(self.dropout),
                    nn.Linear(in_features=hidden_features, out_features=hidden_features),
                    nn.PReLU()
                ])

            self.layers.extend([
                nn.Dropout(self.dropout),
                nn.Linear(in_features=hidden_features, out_features=out_features),
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class RegressorModule(nn.Module):
    def __init__(self, num_layers: int, hidden_features: int, in_features: int,
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        self.dropout = dropout or 0.0

        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise Exception("Number of layers must be at least 1.")

        if num_layers <= 1:
            # Layer from in_features to 1, directly
            self.layers.extend([
                nn.Dropout(self.dropout),
                nn.Linear(in_features=in_features, out_features=1),
            ])
        else:
            # We add layers with hidden_features size
            self.layers.extend([
                nn.Dropout(self.dropout),
                nn.Linear(in_features=in_features, out_features=hidden_features),
                nn.PReLU()
            ])

            for i in range(num_layers - 2):
                self.layers.extend([
                    nn.Dropout(self.dropout),
                    nn.Linear(in_features=hidden_features, out_features=hidden_features),
                    nn.PReLU()
                ])

            self.layers.extend([
                nn.Dropout(self.dropout),
                nn.Linear(in_features=hidden_features, out_features=1),
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


@Attention.register("uniform-attention")
class UniformAttention(Attention):
    """
    Assigns equal weight to all valid rows of the attention function. A sort of ``baseline'' attention
    that just takes the mean of all valid rows.


    Inputs:
    - vector: shape ``(batch_size, embedding_dim)``
    - matrix: shape ``(batch_size, num_rows, embedding_dim)``
    - matrix_mask: shape ``(batch_size, num_rows)``, specifying which rows are just padding.

    Output:
    - attention: shape ``(batch_size, num_rows)``.
    """
    def __init__(self, normalize: bool = True) -> None:
        super().__init__(normalize)

    @overrides
    def forward(
        self, vector: torch.Tensor, matrix: torch.Tensor, matrix_mask: torch.Tensor = None) -> torch.Tensor:
        # We just care about setting the similarities equal wherever there is a valid row.
        # We just use the mask for this.
        similarities = matrix_mask
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities


@Model.register("simple-regressor")
class SimpleRegressor(Model):
    """
    """
    def __init__(self, use_ci:bool, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder,
                 vocab: Vocabulary, predictor_type: str,
                 ci_attention: Attention = None, dropout: Optional[float] = None,
                 normalize_outputs: Optional[Tuple[float, float]] = None,
                 year_range: Optional[Tuple[int, int]] = None,
                 predictor_num_layers: Optional[int] = None, predictor_hidden_features: Optional[int] = None) -> None:
        super().__init__(vocab)
        self.normalize_outputs_mean, self.normalize_outputs_std = normalize_outputs or (0.0, 1.0)

        if predictor_type == "cls" and year_range is None:
            raise Exception("Year ranges need to be specified when predictor is a classifier")

        self.year_min, self.year_max = year_range

        self.predictor_type = predictor_type
        self.use_ci = use_ci
        if self.use_ci:
            print("Using both contextual information and events")
        else:
            print("Using events only")

        self.dropout = dropout or 0.0
        self.predictor_num_layers = predictor_num_layers or 1
        self.predictor_hidden_features = predictor_hidden_features or 300

        self.ci_attention = ci_attention

        self.word_embeddings = word_embeddings
        self.encoder = encoder

        if self.use_ci:
            # We take two vectors as input (encoding of event and of CI sentences)
            predictor_input_dim = encoder.get_output_dim() * 2
        else:
            # We only take one vector as input (encoding of event)
            predictor_input_dim = encoder.get_output_dim()

        if predictor_type == "reg":
            self.vec2year = RegressorModule(num_layers=self.predictor_num_layers,
                                            hidden_features=self.predictor_hidden_features,
                                            in_features=predictor_input_dim, dropout=self.dropout)
        elif predictor_type == "cls":
            self.vec2year = ClassifierModule(num_layers=self.predictor_num_layers,
                                            hidden_features=self.predictor_hidden_features,
                                            in_features=predictor_input_dim, dropout=self.dropout,
                                             out_features=self.year_max - self.year_min + 1)
        else:
            raise Exception("Unknown predictor type: '{}'. Valid options: ['cls', 'reg']".format(predictor_type))

        print("Vocabulary size: {}".format(vocab.get_vocab_size('tokens')))

        self.mae = MeanAbsoluteError()
        self.kendall_tau = KendallTauScore()
        self.exact_match = ExactMatchScore()
        self.under_20y = PercentUnderYearRangeScore(year_range=20)
        self.under_50y = PercentUnderYearRangeScore(year_range=50)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "mae": self.mae.get_metric(reset),
            "kendall_tau": self.kendall_tau.get_metric(reset),
            "exact_match": self.exact_match.get_metric(reset),
            "under_20y": self.under_20y.get_metric(reset),
            "under_50y": self.under_50y.get_metric(reset)
        }

    def forward(self,
                metadata: List[Dict[str, Any]],
                event: Dict[str, torch.Tensor],
                information: Dict[str, torch.Tensor] = None,
                year: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length)
        mask = get_text_field_mask(event)

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        self.word_embeddings.eval()  # This disables dropout in the embeddings, like pretrained BERT embedder
        embeddings = self.word_embeddings(event)

        assert mask.shape[0] == embeddings.shape[0]
        # BERT doesn't enforce this (because of wordpiece tokenization). The mask is word-based, and the embeddings are wordpiece-based.
        # But we then set the encoder to be a BERTPooler and are unaffected by this.
        #assert mask.shape[1] == embeddings.shape[1]
        assert embeddings.shape[2] == self.word_embeddings.get_output_dim()

        # shape: (batch_size, encoder_output_dim)
        if torch.sum(mask).int().item() == 0:
            # There is no event text associated with any of the datapoints, so we return a vector of zeroes
            batch_size = mask.shape[0]
            encoder_out = embeddings.new_zeros((batch_size, self.encoder.get_output_dim()))
        else:
            encoder_out = self.encoder(embeddings, mask)

        assert mask.shape[0] == encoder_out.shape[0]
        assert encoder_out.shape[1] == self.encoder.get_output_dim()

        if self.use_ci:
            # Since the CI is a ListField of TextFields, we need to take care to pass num_wrapping_dim
            # (our tensors have an extra dimension)

            # shape: (batch_size, max_ci_sent_per_inp_in_batch, max_ci_sequence_length, encoder_input_dim)
            embeddings_ci = self.word_embeddings(information, num_wrapping_dims=1)

            assert embeddings_ci.shape[0] == mask.shape[0]
            assert embeddings_ci.shape[3] == self.word_embeddings.get_output_dim()

            # shape: (batch_size, max_ci_sent_per_inp_in_batch, max_ci_sequence_length)
            mask_ci = get_text_field_mask(information, num_wrapping_dims=1)

            assert mask_ci.shape[0] == embeddings_ci.shape[0]
            assert mask_ci.shape[1] == embeddings_ci.shape[1]
            # BERT doesn't enforce this (because of wordpiece tokenization). The mask is word-based, and the embeddings are wordpiece-based.
            # But we then set the encoder to be a BERTPooler and are unaffected by this.
            #assert mask_ci.shape[2] == embeddings_ci.shape[2]

            # shape: (batch_size, encoder_output_dim)
            #
            # Here we need to be careful, as AllenNLP crashes if all datapoints in the batch are devoid of CI.
            # We handle this case by returning a vector of zeroes for all datapoints. This is compatible with how
            # AllenNLP handles the encoding of sequences of length zero (i.e. cases were there is no CI to encode,
            # because of padding or because there just isn't any CI associated with the datapoint).
            if torch.sum(mask_ci).int().item() == 0:
                # There is no CI associated with any of the datapoints. We return a vector of zeroes
                batch_size = mask_ci.shape[0]
                encoder_out_ci_condensed = embeddings_ci.new_zeros((batch_size, self.encoder.get_output_dim()))

                assert encoder_out_ci_condensed.shape[0] == mask_ci.shape[0]
                assert encoder_out_ci_condensed.shape[1] == self.encoder.get_output_dim()
                assert len(encoder_out_ci_condensed.shape) == 2
            else:
                # shape: (batch_size, max_ci_sent_per_inp_in_batch, encoder_output_dim)
                #
                # Given a module and some tensor, TimeDistributed transforms the N-dimensional tensor into an (N-1)-dimensional tensor,
                # by merging the first 2 dimensions, then passes this tensor to the module, and then splits the merged dimension.
                # Hence we can apply our listfield of textfields to our encoder that only takes text fields.
                # TODO: Right now we are using the exact same encoder as for the event. Maybe we should use a different one?
                encoder_out_ci = TimeDistributed(self.encoder)(embeddings_ci, mask_ci)

                assert encoder_out_ci.shape[0] == mask_ci.shape[0]
                assert encoder_out_ci.shape[1] == mask_ci.shape[1]
                assert encoder_out_ci.shape[2] == self.encoder.get_output_dim()

                # We have the encoding for each CI sentence. Now we reduce it to a single vector.
                # We do so by applying an attention mechanism, and then computing a weighted average
                # of the CI encoder results according to the attention weights.
                #
                # If there is no CI associated with a datapoint, we just return a vector of zeros for that CI embedding.
                # We do this by exploiting the fact that the CI encoding when there isn't any CI is just a vector of zeroes.
                # So for these cases we programatically set the attention of the first item to 1, thus giving the zeroes
                # vector complete attention, which after the weighted average means that the weighted vector is still
                # a vector of zeroes.

                # shape: (batch_size, max_ci_sent_per_inp_in_batch)
                encoder_out_ci_valid = mask_ci[:, :, 0]

                assert encoder_out_ci_valid.shape[0] == mask_ci.shape[0]
                assert encoder_out_ci_valid.shape[1] == mask_ci.shape[1]
                assert len(encoder_out_ci_valid.shape) == 2

                # shape: (batch_size, 1, max_ci_sent_per_inp_in_batch)
                # WARNING: When max_ci_sent_inp_in_batch == 1, and attention is linear, this crashes. Probably because
                #          of broadcasting in the wrong direction when calculating attention. This might be an AllenNLP bug.
                #          other forms of attention work fine.
                encoder_out_ci_attention_weights = self.ci_attention(encoder_out, encoder_out_ci, encoder_out_ci_valid).unsqueeze(dim=1)

                assert encoder_out_ci_attention_weights.shape[0] == mask_ci.shape[0]
                assert encoder_out_ci_attention_weights.shape[1] == 1
                assert encoder_out_ci_attention_weights.shape[2] == mask_ci.shape[1]
                assert len(encoder_out_ci_attention_weights.shape) == 3

                # Now we calculate the weighted encoding.
                # shape: (batch_size, encoder_output_dim)
                encoder_out_ci_condensed = encoder_out_ci_attention_weights.bmm(encoder_out_ci).squeeze(dim=1)
                assert encoder_out_ci_condensed.shape[0] == mask_ci.shape[0]
                assert encoder_out_ci_condensed.shape[1] == self.encoder.get_output_dim()
                assert len(encoder_out_ci_condensed.shape) == 2

            # shape: (batch_size, encoder_output_dim * 2)
            encoder_out_ci_event_combined = torch.cat([encoder_out, encoder_out_ci_condensed], dim=1)

            assert encoder_out_ci_event_combined.shape[0] == mask_ci.shape[0]
            assert encoder_out_ci_event_combined.shape[1] == self.encoder.get_output_dim() * 2

            # shape: if "reg" (batch_size,); if "cls" (batch_size, num_classes)
            predicted_output = self.vec2year(encoder_out_ci_event_combined).squeeze(dim=1)
        else:
            # shape: if "reg" (batch_size,); if "cls" (batch_size, num_classes)
            predicted_output = self.vec2year(encoder_out).squeeze(dim=1)

        # Compute the predictions
        if self.predictor_type == "reg":
            # We only need to calculate the predicted years in regression
            # shape: (batch_size,)
            predicted_year = predicted_output * self.normalize_outputs_std + self.normalize_outputs_mean
            predicted_year = predicted_year.float()
            assert predicted_year.shape[0] == mask.shape[0]

            output = {"year": predicted_year}
        elif self.predictor_type == "cls":
            # For classification we compute the logits, and from that the predicted years
            # shape: (batch_size, num_classes)
            predicted_logit = predicted_output

            assert predicted_logit.shape[0] == mask.shape[0]
            assert predicted_logit.shape[1] == self.year_max - self.year_min + 1

            # shape: (batch_size,)
            predicted_year = predicted_logit.argmax(dim=1) + self.year_min
            predicted_year = predicted_year.float()
            assert predicted_year.shape[0] == mask.shape[0]

            output = {"year": predicted_year, "logit": predicted_logit}

        if year is not None:
            # shape: (batch_size,)
            year = year * self.normalize_outputs_std + self.normalize_outputs_mean
            year = year.float()
            assert year.shape[0] == mask.shape[0]

            # Compute loss
            if self.predictor_type == "reg":
                # MSE loss calculation
                output["loss"] = F.mse_loss(predicted_year, year)
            elif self.predictor_type == "cls":
                # Cross-entropy loss calculation
                year_index = year - self.year_min
                year_index = year_index.long()
                output["loss"] = F.cross_entropy(predicted_logit, year_index)

            # Compute metrics
            self.mae(predicted_year, year)
            self.kendall_tau(predicted_year, year)
            self.exact_match(predicted_year, year)
            self.under_20y(predicted_year, year)
            self.under_50y(predicted_year, year)

        return output
