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
from allennlp.data.fields import TextField, MetadataField, IndexField, ArrayField, SpanField, LabelField, ListField, SequenceField, MultiLabelField

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

import pdb

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_SYMBOL = '@@start@@'
END_SYMBOL = '@@end@@'

@DatasetReader.register("msmarco-v21")
class MsMarcoReader(DatasetReader):
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

        num_data = len(dataset['query'])

        logger.info("Reading the dataset")
        for idx, data_spans in zip(range(num_data), span_json):
            answers = dataset['answers'][str(idx)]
            answer = max(answers)

            question = dataset['query'][str(idx)]
            query_id = dataset['query_id'][str(idx)]
            query_type = dataset['query_type'][str(idx)]

            passages_json = dataset['passages'][str(idx)]
            num_passages = len(passages_json)
            passages = [passages_json[i]['passage_text'] for i in range(num_passages)]
            passages_is_selected = [passages_json[i]['is_selected'] for i in range(num_passages)]
            tokenized_passages_list = [self._tokenizer.tokenize(util.normalize_text(p)) for p in passages]
            passages_length = [len(p) for p in tokenized_passages_list]

            normalized_answer = util.normalize_text(answer)
            normalized_question = util.normalize_text(question)

            tokenized_answer = self._tokenizer.tokenize(normalized_answer)
            tokenized_question = self._tokenizer.tokenize(normalized_question)

            question_field = TextField(tokenized_question, self._token_indexers)
            fields = {'question': question_field}

            # negative data: choose three passages closest to the question 
            not_selected_passages = [passages[i] for i in range(num_passages) if passages_is_selected[i] == 0]
            passage_features = self._tfidf.fit_transform(not_selected_passages)
            question_features = self._tfidf.transform([question])
            distances = pairwise_distances(question_features, passage_features, "cosine").ravel()
            sorted_passages = np.lexsort((not_selected_passages, distances))
            for i in range(3):
                #TODO: bug
                passage_idx = sorted_passages[i]
                tokenized_passage = tokenized_passages_list[passage_idx]
                passage_length = len(tokenized_passage)
                passage_field = TextField(tokenized_passage, self._token_indexers)
                fields['passage'] = passage_field
                #fields['no_span'] = IndexField(passage_length*passage_length, None)
                #start_span_field = MultiLabelField([passage_length], skip_indexing=True, num_labels=200)
                #end_span_field = MultiLabelField([passage_length], skip_indexing=True, num_labels=200)
                #combined_span_field = MultiLabelField([passage_length*passage_length], skip_indexing=True, num_labels=40000)
                #fields['span_combined_list'] = combined_span_field
                #fields['start_spans'] = start_span_field
                #fields['end_spans'] = end_span_field
                fields['spans'] = ListField([SpanField(-1, -1, passage_field.empty_field())])
                yield Instance(fields)

            # positive data: choose passage that has correct answer
            if data_spans is not None:
                for passage_spans in data_spans:
                    passage_idx = passage_spans['passage_idx']
                    best_spans = passage_spans['best_spans']
                    if best_spans is not None:
                        if best_spans[0][2] < 0.9:
                            continue
                        tokenized_passage = tokenized_passages_list[passage_idx]
                        passage_length = len(tokenized_passage)
                        passage_field = TextField(tokenized_passage, self._token_indexers)
                        fields['passage'] = passage_field

                        start_span_list = []
                        end_span_list = []
                        combined_span_list = []
                        span_list = []

                        for span in best_spans:
                            #start_span_field = IndexField(int(span[0]), None)
                            #end_span_field = IndexField(int(span[1]), None)
                            span_field = SpanField(int(span[0]), int(span[1]), passage_field)
                            span_list.append(span_field)
                            #start_span_list.append(int(span[0]))
                            #end_span_list.append(int(span[1]))

                            #combined_index = span[0]*passage_length + span[1]
                            #combined_span_field = IndexField(combined_index, None)
                            #combined_span_list.append(combined_index)
                        
                        #start_span_field = MultiLabelField(start_span_list, skip_indexing=True, num_labels=200)
                        #end_span_field = MultiLabelField(end_span_list, skip_indexing=True, num_labels=200)
                        #combined_span_field = MultiLabelField(combined_span_list, skip_indexing=True, num_labels=40000)
                        #fields['span_combined_list'] = combined_span_field
                        #fields['start_spans'] = start_span_field
                        #fields['end_spans'] = end_span_field
                        fields['spans'] = ListField(span_list)
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
