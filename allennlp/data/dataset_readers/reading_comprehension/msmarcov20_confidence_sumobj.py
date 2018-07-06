import json
import logging
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import TextField, MetadataField, IndexField, ArrayField, SpanField, LabelField, ListField

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

import pdb
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = '@@start@@'
END_SYMBOL = '@@end@@'

@DatasetReader.register("msmarcov20-confidence-sumobj-train")
class MsMarcoReaderTrain(DatasetReader):
    """
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 span_file_path: str = None,
                 extraction_model_path: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._span_file_path = span_file_path
        self._extraction_model_path = extraction_model_path
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words="english")

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        span_file = open(self._span_file_path)
        span_json = json.load(span_file)

        logger.info("Reading the dataset")
        for data, best_spans in zip(dataset, span_json):
            answer = data['answers'][0]
            question = data['query']
            well_formed_answer = data['wellFormedAnswers'][0]
            passages_json = data['passages']
            passages = [passages_json[i]['passage_text'] for i in range(len(passages_json))]
            passages_is_selected = [passages_json[i]['is_selected'] for i in range(len(passages_json))]
            
            normalized_answer = util.normalize_text_msmarco(answer)
            tokenized_answer = self._tokenizer.tokenize(normalized_answer)
            # set question field
            normalized_question = util.normalize_text_msmarco(question)
            tokenized_question = self._tokenizer.tokenize(normalized_question)
            question_field = TextField(tokenized_question, self._token_indexers)
            fields = {'question': question_field}
            # skip contexts that have less than 4 paragraphs
            if len(passages) < 4:
                continue
            # only train instances with rouge score larger than 0.9
            if best_spans['score'] > 0.9:
                # rank passsages based on tf-idf score
                passage_features = self._tfidf.fit_transform(passages)
                question_features = self._tfidf.transform([normalized_question])
                distances = pairwise_distances(question_features, passage_features, "cosine").ravel()
                sorted_passages = np.lexsort((passages, distances))
                # choose 4 passages with highest tf-idf score
                selected_passages = []
                ## choose golden passage first
                normalized_passage = util.normalize_text_msmarco(passages[best_spans['passage_idx']])
                tokenized_passage = self._tokenizer.tokenize(normalized_passage)
                passage_field = TextField(tokenized_passage, self._token_indexers)
                selected_passages.append(passage_field)
                ## set span field from golden passage
                span_start_list = []
                span_end_list = []
                for span in best_spans['best_spans']:
                    span_start_field = IndexField(int(span[0]), passage_field)
                    span_end_field = IndexField(int(span[1]), passage_field)
                    span_start_list.append(span_start_field)
                    span_end_list.append(span_end_field)
                fields['span_start'] = ListField(span_start_list)
                fields['span_end'] = ListField(span_end_list)
                ## choose three others with highest tf-idf score
                idx = 0
                while len(selected_passages) < 4:
                    if sorted_passages[idx] != best_spans['passage_idx']:
                        normalized_passage = util.normalize_text_msmarco(passages[sorted_passages[idx]])
                        tokenized_passage = self._tokenizer.tokenize(normalized_passage)
                        passage_field = TextField(tokenized_passage, self._token_indexers)
                        selected_passages.append(passage_field)
                    idx += 1
                fields['passage'] = ListField(selected_passages)
                yield Instance(fields)

    @overrides
    def text_to_instance(self,  # type: ignore
            answer: str,
            question: str,
            passages: List[str],
            passages_length: List[int],
            passages_is_selected: List[int],
            concatenated_passage: str) -> Instance:
        return None

    @classmethod
    def from_params(cls, params: Params) -> 'MsMarcoReaderTrain':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        span_file_path = params.pop('span_file_path', None)
        extraction_model_path = params.pop('extraction_model_path', None)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   span_file_path=span_file_path,
                   extraction_model_path=extraction_model_path,
                   lazy=lazy)


@DatasetReader.register("msmarcov20-confidence-sumobj-test")
class MsMarcoReaderTest(DatasetReader):
    """
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 span_file_path: str = None,
                 extraction_model_path: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._span_file_path = span_file_path
        self._extraction_model_path = extraction_model_path

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        span_file = open(self._span_file_path)

        logger.info("Reading the dataset")
        for data, best_span in zip(dataset, span_file):
            answer = data['answers'][0]
            question = data['query']
            well_formed_answer = data['wellFormedAnswers'][0]
            passages_json = data['passages']
            passages = [passages_json[i]['passage_text'] for i in range(len(passages_json))]
            passages_is_selected = [passages_json[i]['is_selected'] for i in range(len(passages_json))]
            # normalize answer text
            normalized_answer = util.normalize_text_msmarco(answer)
            # set question field
            normalized_question = util.normalize_text_msmarco(question)
            tokenized_question = self._tokenizer.tokenize(normalized_question)
            question_field = TextField(tokenized_question, self._token_indexers)
            fields = {'question': question_field}
            # set passage field
            normalized_passages = [util.normalize_text_msmarco(p) for p in passages]
            tokenized_passages = [self._tokenizer.tokenize(p) for p in normalized_passages] 
            passage_fields = []
            for tokenized_passage in tokenized_passages:
                passage_field = TextField(tokenized_passage, self._token_indexers)
                passage_fields.append(passage_field)
            fields['passage'] = ListField(passage_fields)
            # set metadata
            metadata = {
                'question_tokens': [token.text for token in tokenized_question],
                'passage_offsets': [[(token.idx, token.idx + len(token.text)) for token in passage_field.tokens] for passage_field in passage_fields],
                'all_passages': normalized_passages,
                'answer_texts': [normalized_answer]
                }
            fields['metadata'] = MetadataField(metadata)
            yield Instance(fields)

    @overrides
    def text_to_instance(self,  # type: ignore
            answer: str,
            question: str,
            passages: List[str],
            passages_length: List[int],
            passages_is_selected: List[int],
            concatenated_passage: str) -> Instance:
        return None

    @classmethod
    def from_params(cls, params: Params) -> 'MsMarcoReaderTest':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        span_file_path = params.pop('span_file_path', None)
        extraction_model_path = params.pop('extraction_model_path', None)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer,
                   token_indexers=token_indexers,
                   span_file_path=span_file_path,
                   extraction_model_path=extraction_model_path,
                   lazy=lazy)
