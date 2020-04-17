from typing import List, Optional, Dict
from overrides import overrides

from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data.instance import Instance

import copy
import numpy as np


@Predictor.register("heo-predictor")
class HEOPredictor(Predictor):
    """
    For obtaining HEO predictions.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        event = json_dict["Event"]
        if "Information" in json_dict:
            information = json_dict["Information"]

        return self._dataset_reader.text_to_instance(event=event, information=information)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = super().predict_instance(instance)
        outputs["event"] = instance["metadata"]["event"]
        outputs["information"] = instance["metadata"]["information"]

        if "year" in instance["metadata"]:
            outputs["target_year"] = instance["metadata"]["year"]

        return sanitize(outputs)
