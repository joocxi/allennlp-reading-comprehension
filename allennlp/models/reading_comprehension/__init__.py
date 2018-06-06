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
from allennlp.models.reading_comprehension.snet_fix import EvidenceExtractionFix
from allennlp.models.reading_comprehension.modelv21 import ModelV21
from allennlp.models.reading_comprehension.model_for_squad import ModelSQUAD
from allennlp.models.reading_comprehension.bidaf_self_attention import BidafPlusSelfAttention
