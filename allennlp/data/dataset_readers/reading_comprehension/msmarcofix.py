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

#from allennlp.models.archival import load_archive

import pdb

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = '@@start@@'
END_SYMBOL = '@@end@@'

@DatasetReader.register("msmarcofix")
class MsMarcoReader(DatasetReader):
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

            tokenized_passages_list = [self._tokenizer.tokenize(util.normalize_text(p)) for p in passages]
            passages_length = [len(p) for p in tokenized_passages_list]
            #cumulative_passages_length = np.cumsum(passages_length)

            normalized_answer = util.normalize_text(answer)
            normalized_question = util.normalize_text(question)

            tokenized_answer = self._tokenizer.tokenize(normalized_answer)
            tokenized_question = self._tokenizer.tokenize(normalized_question)

            question_field = TextField(tokenized_question, self._token_indexers)
            fields = {'question': question_field}

            start_idx, end_idx, rouge_score, passage_idx = None, None, None, None

            tokenized_passage = [token for sublist in tokenized_passages_list for token in sublist]
            start_idx, end_idx, passage_idx, rouge_score = best_span.strip().split(' ')
            start_idx, end_idx, passage_idx, rouge_score = int(start_idx), int(end_idx), int(passage_idx), float(rouge_score)

            if start_idx + 5 > end_idx:
                continue
            if rouge_score > 0.7:
                passage_field = TextField(tokenized_passage, self._token_indexers)
                fields['passage'] = passage_field

                span_start_field = IndexField(start_idx, None)
                span_end_field = IndexField(end_idx, None)

                fields['passages_length'] = ArrayField(np.asarray(passages_length))
                fields['span_start'] = span_start_field
                fields['span_end'] = span_end_field
                #list = []
                #for i in range(len(passages_length)):
                #    list.append(TextField(tokenized_passages_list[i], self._token_indexers))
                #fields['passages_list'] = ListField(list) 

                # TODO:
                correct_passage_field = LabelField(passage_idx, skip_indexing=True)
                fields['correct_passage'] = correct_passage_field
                yield Instance(fields)

    @overrides
    def text_to_instance(self,  # type: ignore
            answer: str,
            question: str,
            passages: List[str],
            passages_length: List[int],
            passages_is_selected: List[int],
            concatenated_passage: str) -> Instance:

        passage_field = TextField(tokenized_passage, self._token_indexers)

        passages_length_field = ArrayField(np.asaray(passages_length))
        passages_is_selected_field = ArrayField(np.asarray(passages_is_selected))

        fields = {'answer': answer_field, 'question': question_field, 'passage': passage_field,
                  'passages_length': passages_length_field, 'passages_is_selected': passages_is_selected_field}

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'MsMarcoReader':
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

@DatasetReader.register("my-msmarco2-fix")
class MSMARCOPassageReader2(DatasetReader):
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

        # if self._span_file_path is not None:
        span_file = open(self._span_file_path)

        span_file = json.load(span_file)
        #archive = load_archive(self._extraction_model_path)
        #model = archive.model
        model = None
        p1_dataset_reader = DatasetReader.from_params(archive.config["dataset_reader"])
        p1_token_indexers = p1_dataset_reader._token_indexers

        logger.info("Reading the dataset")
        for data, best_span in zip(dataset, span_file):
            answer = data['answers'][0]
            question = data['query']
            well_formed_answer = data['wellFormedAnswers'][0]
            passages_json = data['passages']
            passages = [passages_json[i]['passage_text'] for i in range(len(passages_json))]
            # passages_length = [len(p) for p in passages]
            passages_is_selected = [passages_json[i]['is_selected'] for i in range(len(passages_json))]
            # concatenated_passage = ' '.join(passages)
            tokenized_passages_list = [self._tokenizer.tokenize(util.normalize_text(p)) for p in passages]
            passages_length = [len(p) for p in tokenized_passages_list]
            cumulative_passages_length = np.cumsum(passages_length)

            normalized_answer =  None
            if answer != None:
                normalized_answer = util.normalize_text(answer)
            normalized_question = util.normalize_text(question)

            tokenized_answer = self._tokenizer.tokenize(normalized_answer)
            tokenized_question = self._tokenizer.tokenize(normalized_question)

            question_field = TextField(tokenized_question, self._token_indexers)
            fields = {'question': question_field}

            start_idx, end_idx, rouge_score, passage_idx = None, None, None, None

            tokenized_answer.insert(0, Token(START_SYMBOL))
            tokenized_answer.append(Token(END_SYMBOL))
            tokenized_passage = [token for sublist in tokenized_passages_list for token in sublist]
            passage_field = TextField(tokenized_passage, self._token_indexers)
            fields['passage'] = passage_field

            p1_question_field = TextField(tokenized_question, p1_token_indexers)
            p1_passage_field = TextField(tokenized_passage, p1_token_indexers)
            p1_fields = {'question': p1_question_field, 'passage': p1_passage_field}
            p1_instance = Instance(p1_fields)
            outputs = model.forward_on_instance(p1_instance, -1)

            start_idx = outputs['span_start_idx']
            end_idx = outputs['span_end_idx']
            for idx in range(len(cumulative_passages_length)):
                if start_idx < cumulative_passages_length[idx]:
                    break

            if idx != 0:
                start_idx = start_idx - cumulative_passages_length[idx -  1]
                end_idx = end_idx - cumulative_passages_length[idx - 1]

            assert start_idx <= end_idx, "Span prediction does not make sense!!!"
            
            # yield instance from predicted span
            span_start_field = IndexField(int(start_idx), passage_field)
            span_end_field = IndexField(int(end_idx), passage_field)
            answer_field = TextField(tokenized_answer, self._token_indexers)

            fields['passage'] = passage_field
            fields['span_start'] = span_start_field
            fields['span_end'] = span_end_field
            fields['answer'] = answer_field

            evidence = self.get_evidence(tokenized_passage, int(start_idx), int(end_idx))
            fields['metadata'] = MetadataField({
                'evidence': evidence,
                'question_text': normalized_question,
                'answer_text': normalized_answer
            })

            yield Instance(fields)

            # yield instances from gold spans
            for item in best_span:
                if item['score'] > 0.5:
                    passage_field = TextField(tokenized_passages_list[item['passage']], self._token_indexers)
                    span_start_field = IndexField(item['start'], passage_field)
                    span_end_field = IndexField(item['end'], passage_field)
                    answer_field = TextField(tokenized_answer, self._token_indexers)

                    fields['passage'] = passage_field
                    fields['span_start'] = span_start_field
                    fields['span_end'] = span_end_field
                    fields['answer'] = answer_field

                    evidence = self.get_evidence(tokenized_passages_list[item['passage']], int(start_idx), int(end_idx))
                    fields['metadata'] = MetadataField({
                        'evidence': evidence,
                        'question_text': normalized_question,
                        'answer_text': normalized_answer
                    })

                    yield Instance(fields)

    def get_evidence(self, passage: List[Token], start_idx: int, end_idx: int):
        lst = [token.text for token in passage[start_idx: end_idx + 1]]
        return ' '.join(lst)

    @overrides
    def text_to_instance(self,  # type: ignore
            answer: str,
            question: str,
            passages: List[str],
            passages_length: List[int],
            passages_is_selected: List[int],
            concatenated_passage: str) -> Instance:

        passage_field = TextField(tokenized_passage, self._token_indexers)

        passages_length_field = ArrayField(np.asaray(passages_length))
        passages_is_selected_field = ArrayField(np.asarray(passages_is_selected))

        fields = {'answer': answer_field, 'question': question_field, 'passage': passage_field,
                  'passages_length': passages_length_field, 'passages_is_selected': passages_is_selected_field}

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'MSMARCOPassageReader2':
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
