from typing import Dict, List, Optional, Any, Tuple
from overrides import overrides

import pandas as pd
import numpy as np
import dateutil.parser as parser
import datetime

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.fields import Field, TextField, ListField, MetadataField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer, SpacyWordSplitter


@DatasetReader.register("heo-reader")
class HEODatasetReader(DatasetReader):
    def __init__(self, token_indexers: Optional[Dict[str, TokenIndexer]] = None, tokenizer: Optional[Tokenizer] = None,
                 limit_number: Optional[int] = None, normalize_outputs: Optional[Tuple[float, float]] = None, lazy: bool = False) -> None:
        super().__init__(lazy)

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer()
        self._limit_number = limit_number

        self._normalize_outputs_mean, self._normalize_outputs_std = normalize_outputs or (0.0, 1.0)

        self._dummy_text_field = TextField([Token("foo")], self._token_indexers)

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        i = 0
        data = pd.read_pickle(file_path)
        for idx, row in data.iterrows():
            if self._limit_number and i >= self._limit_number:
                raise StopIteration()

            i += 1

            yield self.create_instance_from_row(row)

    def create_instance_from_row(self, row):
        event = row["Event"]
        information = row["Information"]

        if "YY" in row:
            year = int(row["YY"])
            return self.text_to_instance(event, information, year)
        else:
            return self.text_to_instance(event, information)

    def tokenize_event(self, event: str):
        return self._tokenizer.tokenize(event)

    def tokenize_ci(self, ci: str):
        return self._tokenizer.tokenize(ci)

    @overrides
    def text_to_instance(self, event: str, information: List[str], year: Optional[int] = None) -> Instance:
        fields: Dict[str, Field] = {}

        # Prepare event field
        tokens = TextField(self.tokenize_event(event), self._token_indexers)
        fields["event"] = tokens

        # Prepare information field
        if information is not None and len(information) > 0:
            info = ListField([TextField(self.tokenize_ci(i), self._token_indexers) for i in information])
        else:
            info = ListField([self._dummy_text_field.empty_field()])

        fields["information"] = info

        metadata: Dict[str, Any] = {}
        metadata["event"] = event
        metadata["information"] = information
        if year:
            metadata["year"] = year

        fields["metadata"] = MetadataField(metadata)

        if year is not None:
            fields["year"] = ArrayField(np.array((float(year) - self._normalize_outputs_mean) / self._normalize_outputs_std))

        return Instance(fields)


@DatasetReader.register("heo-reader-only-events-with-ci")
class HEODatasetReaderOnlyCI(HEODatasetReader):
    """ Like HEODatasetReader, but ignores all events that do not have CI. """
    def __init__(self, no_event: bool, token_indexers: Optional[Dict[str, TokenIndexer]] = None, tokenizer: Optional[Tokenizer] = None,
                 limit_number: Optional[int] = None, normalize_outputs: Optional[Tuple[float, float]] = None, lazy: bool = False) -> None:
        super().__init__(token_indexers, tokenizer, limit_number, normalize_outputs, lazy)

        self._no_event = no_event

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        i = 0
        data = pd.read_pickle(file_path)
        for idx, row in data.iterrows():
            if self._limit_number and i >= self._limit_number:
                raise StopIteration()

            # Skip rows if they do not have CI--this is the purpose of this dataset reader
            if row["Information"] is not None and len(row["Information"]) > 0:
                i += 1

                yield self.create_instance_from_row(row)

    @overrides
    def tokenize_event(self, event: str):
        if self._no_event:
            return self._dummy_text_field.empty_field()
        else:
            return self._tokenizer.tokenize(event)


@DatasetReader.register("heo-reader-ablate-tokens")
class HEODatasetReaderAblateTokens(HEODatasetReader):
    """ Like HEODatasetReader, but ignores all events that do not have CI. """
    def __init__(self, ablate_mode: str,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None, tokenizer: Optional[Tokenizer] = None,
                 limit_number: Optional[int] = None, normalize_outputs: Optional[Tuple[float, float]] = None, lazy: bool = False) -> None:
        super().__init__(token_indexers, tokenizer, limit_number, normalize_outputs, lazy)

        assert ablate_mode in ["years", "dates", "numbers"]
        self._ablate_mode = ablate_mode

        # Ensure tokenizer creates the tags needed for filtering
        # Since we may need to use spaCy's `like_num` property that is not inherited by AlleNLP tokens, we keep the spacy tokens directly
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True, ner=True, keep_spacy_tokens=True))

    def apply_filters(self, tokens: str):
        if self._ablate_mode == "years":
            tokens = self.remove_year_tokens(tokens)
        elif self._ablate_mode == "dates":
            tokens = self.remove_date_tokens(tokens)
        elif self._ablate_mode == "numbers":
            tokens = self.remove_number_tokens(tokens)

        return tokens

    @overrides
    def tokenize_event(self, event: str):
        tokens = self._tokenizer.tokenize(event)
        return self.apply_filters(tokens)

    @overrides
    def tokenize_ci(self, ci: str):
        tokens = self._tokenizer.tokenize(ci)
        return self.apply_filters(tokens)

    def remove_year_tokens(self, tokens):
        current_year = datetime.datetime.now().year
        mask = [True] * len(tokens)

        for idx, token in enumerate(tokens):
            if token.ent_type_ == "DATE":
                if token.pos_ == "NUM":
                    try:
                        year = parser.parse(token.text).year
                        if year < current_year:
                            mask[idx] = False
                        elif year == current_year and str(current_year) in token.text:
                            # Needed due to problems with small numbers that are not years
                            # Dateutil parser returns current year if the number was actually another thing
                            mask[idx] = False
                    except Exception:
                        continue

        # Apply mask
        zipped = zip(mask, tokens)
        final = [tok for (m, tok) in zipped if m]

        return final

    def remove_date_tokens(self, tokens):
        return [x for x in tokens if x.ent_type_ != "DATE"]

    def remove_number_tokens(self, tokens):
        return [x for x in tokens if not x.like_num]
