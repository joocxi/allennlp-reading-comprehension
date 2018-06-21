"""
Reading comprehension is loosely defined as follows: given a question and a passage of text that
contains the answer, answer the question.

These submodules contain readers for things that are predominantly reading comprehension datasets.
"""

from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data.dataset_readers.reading_comprehension.triviaqa import TriviaQaReader
from allennlp.data.dataset_readers.reading_comprehension.my_msmarco import MsMarcoReader
from allennlp.data.dataset_readers.reading_comprehension.msmarcofix import MsMarcoReader
from allennlp.data.dataset_readers.reading_comprehension.msmarcov21 import MsMarcoReader
from allennlp.data.dataset_readers.reading_comprehension.msmarcov20_single_paragraph import MsMarcoReaderTrain, MsMarcoReaderTest
from allennlp.data.dataset_readers.reading_comprehension.msmarcov20_confidence import MsMarcoReaderTrain, MsMarcoReaderTest
from allennlp.data.dataset_readers.reading_comprehension.multi_squad_train import SquadReader
from allennlp.data.dataset_readers.reading_comprehension.multi_squad_dev import SquadReader

