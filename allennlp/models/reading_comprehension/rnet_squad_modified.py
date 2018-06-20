import logging
from typing import Any, Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.models.model import Model
from allennlp.modules import Attention, FeedForward, Maxout
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder 
from allennlp.modules.similarity_functions import NonlinearSimilarity
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import SquadEmAndF1, CategoricalAccuracy, BooleanAccuracy
from allennlp.data.dataset_readers.reading_comprehension.my_msmarco import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
ITE = 0

@Model.register("rnet-squad-mod")
class EvidenceExtraction(Model):
    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 question_encoder: Seq2SeqEncoder,
                 passage_encoder: Seq2SeqEncoder,
                 r: float = 0.8,
                 dropout: float = 0.1,
		 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EvidenceExtraction, self).__init__(vocab, regularizer)

        self._embedder = embedder

        self._question_encoder = question_encoder
        self._passage_encoder = passage_encoder

        # size: 2H
        encoding_dim = question_encoder.get_output_dim()

        self._gru_cell = nn.GRUCell(2*encoding_dim, encoding_dim)

        self._gate = nn.Linear(2*encoding_dim, 2*encoding_dim)

        self._match_layer_1 = nn.Linear(2*encoding_dim, encoding_dim)
        self._match_layer_2 = nn.Linear(encoding_dim, 1)

        self._question_attention_for_passage = Attention(NonlinearSimilarity(encoding_dim))
        self._question_attention_for_question = Attention(NonlinearSimilarity(encoding_dim))
        self._passage_attention_for_answer = Attention(NonlinearSimilarity(encoding_dim), normalize=False)
        self._passage_attention_for_ranking = Attention(NonlinearSimilarity(encoding_dim))

        self._passage_self_attention = Attention(NonlinearSimilarity(encoding_dim))
        self._self_gru_cell = nn.GRUCell(2*encoding_dim, encoding_dim)
        self._self_gate = nn.Linear(2*encoding_dim, encoding_dim) 

        self._answer_net = nn.GRUCell(encoding_dim, encoding_dim)

        self._v_r_Q = nn.Parameter(torch.rand(encoding_dim)) 
        self._r = r

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializer(self)

    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata = None) -> Dict[str, torch.Tensor]:
     
        # shape: B x Tq x E
        embedded_question = self._embedder(question)
        embedded_passage = self._embedder(passage)

        batch_size = embedded_question.size(0)
        total_passage_length = embedded_passage.size(1)

        question_mask = util.get_text_field_mask(question)
        passage_mask = util.get_text_field_mask(passage)

        # shape: B x T x 2H
        encoded_question = self._dropout(self._question_encoder(embedded_question, question_mask))
        encoded_passage = self._dropout(self._passage_encoder(embedded_passage, passage_mask))
        passage_mask = passage_mask.float()
        question_mask = question_mask.float()

        encoding_dim = encoded_question.size(-1)

        # shape: B x 2H
        if encoded_passage.is_cuda:
            cuda_device = encoded_passage.get_device()
            gru_hidden = Variable(torch.zeros(batch_size, encoding_dim).cuda(cuda_device))
        else:
            gru_hidden = Variable(torch.zeros(batch_size, encoding_dim))

        question_awared_passage = []
        for timestep in range(total_passage_length):
            # shape: B x Tq = attention(B x 2H, B x Tq x 2H)
            attn_weights = self._question_attention_for_passage(encoded_passage[:, timestep, :], encoded_question, question_mask)
            # shape: B x 2H = weighted_sum(B x Tq x 2H, B x Tq)
            attended_question = util.weighted_sum(encoded_question, attn_weights)
            # shape: B x 4H
            passage_question_combined = torch.cat([encoded_passage[:, timestep, :], attended_question], dim=-1)
            # shape: B x 4H
            gate = F.sigmoid(self._gate(passage_question_combined))
            gru_input = gate * passage_question_combined
            # shape: B x 2H
            gru_hidden = self._dropout(self._gru_cell(gru_input, gru_hidden))
            question_awared_passage.append(gru_hidden)

        # shape: B x T x 2H
        # question aware passage representation v_P
        question_awared_passage = torch.stack(question_awared_passage, dim=1)

        self_attended_passage = []
        for timestep in range(total_passage_length):
            attn_weights = self._passage_self_attention(question_awared_passage[:, timestep, :], question_awared_passage, passage_mask)
            attended_passage = util.weighted_sum(question_awared_passage, attn_weights)
            input_combined = torch.cat([question_awared_passage[:, timestep, :], attended_passage], dim=-1)
            gate = F.sigmoid(self._self_gate(input_combined))
            gru_input = gate * input_combined
            gru_hidden = self._dropout(self._gru_cell(gru_input, gru_hidden))
            self_attended_passage.append(gru_hidden)

        self_attended_passage = torch.stack(self_attended_passage, dim=1)

        # compute question vector r_Q
        # shape: B x T = attention(B x 2H, B x T x 2H)
        v_r_Q_tiled = self._v_r_Q.unsqueeze(0).expand(batch_size, encoding_dim)
        attn_weights = self._question_attention_for_question(v_r_Q_tiled, encoded_question, question_mask)
        # shape: B x 2H
        r_Q = util.weighted_sum(encoded_question, attn_weights)
        # shape: B x T = attention(B x 2H, B x T x 2H)
        span_start_logits = self._passage_attention_for_answer(r_Q, self_attended_passage, passage_mask)
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)
        span_start_log_probs = util.masked_log_softmax(span_start_logits, passage_mask)
        # shape: B x 2H
        c_t = util.weighted_sum(self_attended_passage, span_start_probs)
        # shape: B x 2H
        h_1 = self._dropout(self._answer_net(c_t, r_Q))

        span_end_logits = self._passage_attention_for_answer(h_1, self_attended_passage, passage_mask)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        span_end_log_probs = util.masked_log_softmax(span_end_logits, passage_mask)
        
        best_span = self.get_best_span(span_start_logits, span_end_logits)

        #num_passages = passages_length.size(1)
        #acc = Variable(torch.zeros(batch_size, num_passages + 1)).cuda(cuda_device).long()

        #acc[:, 1:num_passages+1] = torch.cumsum(passages_length, dim=1)

        #g_batch = []
        #for b in range(batch_size):
        #    g = []
        #    for i in range(num_passages):
        #        if acc[b, i+1].data[0] > acc[b, i].data[0]:
        #            attn_weights = self._passage_attention_for_ranking(r_Q[b:b+1], question_awared_passage[b:b+1, acc[b, i].data[0]: acc[b, i+1].data[0], :], passage_mask[b:b+1, acc[b, i].data[0]: acc[b, i+1].data[0]])
        #            r_P = util.weighted_sum(question_awared_passage[b:b+1, acc[b, i].data[0]:acc[b, i+1].data[0], :], attn_weights)
        #            question_passage_combined = torch.cat([r_Q[b:b+1], r_P], dim=-1)
        #            gi = self._dropout(self._match_layer_2(F.tanh(self._dropout(self._match_layer_1(question_passage_combined)))))
        #            g.append(gi)
        #        else:
        #            g.append(Variable(torch.zeros(1, 1)).cuda(cuda_device))
        #    g = torch.cat(g, dim=1)
        #    g_batch.append(g)
        
        #t2 = time.time()
        #g = torch.cat(g_batch, dim=0)
        output_dict = {}
        if span_start is not None:
            AP_loss = F.nll_loss(span_start_log_probs, span_start.squeeze(-1)) +\
                F.nll_loss(span_end_log_probs, span_end.squeeze(-1))
            #PR_loss = F.nll_loss(passage_log_probs, correct_passage.squeeze(-1))
            #loss = self._r * AP_loss + self._r * PR_loss
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            output_dict['loss'] = AP_loss

        _, max_start = torch.max(span_start_probs, dim=1)
        _, max_end = torch.max(span_end_probs, dim=1)
        #t3 = time.time()
        output_dict['span_start_idx'] = max_start
        output_dict['span_end_idx'] = max_end
        #t4 = time.time()
        #global ITE
        #ITE += 1
        #if (ITE % 100 == 0):
        #    print(" gold %i:%i|predicted %i:%i" %(span_start.squeeze(-1)[0], span_end.squeeze(-1)[0], max_start.data[0], max_end.data[0]))
        if metadata is not None:
             output_dict['best_span_str'] = []
             question_tokens = []
             passage_tokens = []
             for i in range(batch_size): 
                 question_tokens.append(metadata[i]['question_tokens'])
                 passage_tokens.append(metadata[i]['passage_tokens']) 
                 passage_str = metadata[i]['original_passage']
                 offsets = metadata[i]['token_offsets']
                 predicted_span = tuple(best_span[i].data.cpu().numpy())
                 start_offset = offsets[predicted_span[0]][0]
                 end_offset = offsets[predicted_span[1]][1]
                 best_span_string = passage_str[start_offset:end_offset]
                 output_dict['best_span_str'].append(best_span_string)
                 answer_texts = metadata[i].get('answer_texts', [])
                 if answer_texts:
                     self._squad_metrics(best_span_string, answer_texts)
             output_dict['question_tokens'] = question_tokens
             output_dict['passage_tokens'] = passage_tokens
        
        #t5 = time.time()
        #print("Total: %.5f" % (t5-t0))
        #print("Batch processing 1: %.5f" % (t2-t1))
        #print("Batch processing 2: %.5f" % (t4-t3))
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score
                }

    @staticmethod
    def get_best_span(span_start_logits: Variable, span_end_logits: Variable) -> Variable:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j 
                    val1 = span_start_logits[b, j] 
                val2 = span_end_logits[b, j]
                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2 
        return best_word_span  

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'EvidenceExtraction':
        embedder_params = params.pop("text_field_embedder")
        embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        question_encoder = Seq2SeqEncoder.from_params(params.pop("question_encoder"))
        passage_encoder = Seq2SeqEncoder.from_params(params.pop("passage_encoder"))
        dropout = params.pop_float('dropout', 0.1)
        r = params.pop_float('r', 0.8)
        #cuda = params.pop_int('cuda', 0)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   embedder=embedder,
                   question_encoder=question_encoder,
                   passage_encoder=passage_encoder,
                   r=r,
                   dropout=dropout,
                   #cuda=cuda,
                   initializer=initializer,
                   regularizer=regularizer)
