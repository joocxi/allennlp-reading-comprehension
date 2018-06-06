"""
Reading comprehension is loosely defined as follows: given a question and a passage of text that
contains the answer, answer the question.

These submodules contain readers for things that are predominantly reading comprehension datasets.
"""

from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader
from allennlp.data.dataset_readers.reading_comprehension.triviaqa import TriviaQaReader
from allennlp.data.dataset_readers.reading_comprehension.my_msmarco import MSMARCOPassageReader
from allennlp.data.dataset_readers.reading_comprehension.msmarcofix import MSMARCOPassageReaderFix
from allennlp.data.dataset_readers.reading_comprehension.msmarcov21 import MSMARCOV21

