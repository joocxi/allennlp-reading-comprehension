import json
import logging
import numpy as np
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

import pdb

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = '@@start@@'
END_SYMBOL = '@@end@@'

@DatasetReader.register("msmarcov20-single-paragraph-train")
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
            #well_formed_answer = data['wellFormedAnswers'][0]
            passages_json = data['passages']
            passages = [passages_json[i]['passage_text'] for i in range(len(passages_json))]
            passages_is_selected = [passages_json[i]['is_selected'] for i in range(len(passages_json))]

            normalized_answer = util.normalize_text_msmarco(answer)
            normalized_question = util.normalize_text_msmarco(question)
    
            #tokenized_answer = self._tokenizer.tokenize(normalized_answer)
            tokenized_question = self._tokenizer.tokenize(normalized_question)

            question_field = TextField(tokenized_question, self._token_indexers)
            fields = {'question': question_field}

            start_idx, end_idx, rouge_score, passage_idx = None, None, None, None
            start_idx, end_idx, passage_idx, rouge_score = best_span.strip().split(' ')
            start_idx, end_idx, passage_idx, rouge_score = int(start_idx), int(end_idx), int(passage_idx), float(rouge_score)

            if start_idx + 1 > end_idx:
                continue
            if rouge_score > 0.9:
                normalized_passage = util.normalize_text_msmarco(passages[passage_idx])
                tokenized_passage = self._tokenizer.tokenize(normalized_passage)
                passage_field = TextField(tokenized_passage, self._token_indexers)
                fields['passage'] = passage_field

                span_start_field = IndexField(start_idx, passage_field)
                span_end_field = IndexField(end_idx, passage_field)

                fields['span_start'] = span_start_field
                fields['span_end'] = span_end_field
                
                passage_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenized_passage]

                metadata = {
                        'original_passage': normalized_passage,
                        'token_offsets': passage_offsets,
                        'question_tokens': [token.text for token in tokenized_question],
                        'passage_tokens': [token.text for token in tokenized_passage]
                        }
                
                if answer:
                    metadata['answer_texts'] = [normalized_answer] 

                fields['metadata'] = MetadataField(metadata)
                yield Instance(fields)


    @overrides
    def text_to_instance(self) -> Instance:  # type: ignore
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


@DatasetReader.register("msmarcov20-single-paragraph-test")
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

        logger.info("Reading the dataset")
        for data in dataset:
            answer = data['answers'][0]
            question = data['query']
            #well_formed_answer = data['wellFormedAnswers'][0]
            passages_json = data['passages']
            passages = [passages_json[i]['passage_text'] for i in range(len(passages_json))]
            passages_is_selected = [passages_json[i]['is_selected'] for i in range(len(passages_json))]

            normalized_answer = util.normalize_text_msmarco(answer)
            normalized_question = util.normalize_text_msmarco(question)
    
            #tokenized_answer = self._tokenizer.tokenize(normalized_answer)
            tokenized_question = self._tokenizer.tokenize(normalized_question)

            question_field = TextField(tokenized_question, self._token_indexers)
            fields = {'question': question_field}

            for passage_idx, is_selected in enumerate(passages_is_selected): 
                if is_selected:
                    normalized_passage = util.normalize_text_msmarco(passages[passage_idx])
                    tokenized_passage = self._tokenizer.tokenize(normalized_passage)
                    passage_field = TextField(tokenized_passage, self._token_indexers)
                    fields['passage'] = passage_field
                
                    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenized_passage]

                    metadata = {
                        'original_passage': normalized_passage,
                        'token_offsets': passage_offsets,
                        'question_tokens': [token.text for token in tokenized_question],
                        'passage_tokens': [token.text for token in tokenized_passage]
                        }
                
                    if answer:
                        metadata['answer_texts'] = [normalized_answer] 

                    fields['metadata'] = MetadataField(metadata)
                    yield Instance(fields)
                    break


    @overrides
    def text_to_instance(self) -> Instance:  # type: ignore
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
