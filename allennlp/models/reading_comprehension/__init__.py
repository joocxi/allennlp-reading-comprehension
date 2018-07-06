"""
Reading comprehension is loosely defined as follows: given a question and a passage of text that
contains the answer, answer the question.

These submodules contain models for things that are predominantly focused on reading comprehension.
"""

from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.models.reading_comprehension.bidaf_copy import BidirectionalAttentionFlow
from allennlp.models.reading_comprehension.bidaf_ensemble import BidafEnsemble
from allennlp.models.reading_comprehension.bidaf_ensemble import BidafEnsemble
from allennlp.models.reading_comprehension.snet import EvidenceExtraction
from allennlp.models.reading_comprehension.snet_fix import EvidenceExtraction
from allennlp.models.reading_comprehension.snet_squad import EvidenceExtraction
from allennlp.models.reading_comprehension.rnet_squad import EvidenceExtraction
from allennlp.models.reading_comprehension.modelv21 import ModelV21
from allennlp.models.reading_comprehension.model_squad import ModelSQUAD
from allennlp.models.reading_comprehension.confidence_model_for_squad import ModelSQUAD
from allennlp.models.reading_comprehension.model_squad_sharenorm import ModelSQUAD
from allennlp.models.reading_comprehension.model_squad_noanswer import ModelSQUAD
from allennlp.models.reading_comprehension.model_msmarcov20 import ModelMSMARCO
from allennlp.models.reading_comprehension.model_msmarcov20_sumobj import ModelMSMARCO
from allennlp.models.reading_comprehension.model_msmarcov20_sharenorm import ModelMSMARCO
from allennlp.models.reading_comprehension.model_msmarcov20_sharenorm_sumobj import ModelMSMARCO
