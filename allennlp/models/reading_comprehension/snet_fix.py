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
from allennlp.training.metrics import SquadEmAndF1, CategoricalAccuracy
from allennlp.data.dataset_readers.reading_comprehension.my_msmarco import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
ITE = 0

@Model.register("snet-fix")
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

        self._gru_cell = nn.GRUCell(2 * encoding_dim, encoding_dim)

        self._gate = nn.Linear(2 * encoding_dim, 2 * encoding_dim)

        self._match_layer_1 = nn.Linear(2 * encoding_dim, encoding_dim)
        self._match_layer_2 = nn.Linear(encoding_dim, 1)

        self._question_attention_for_passage = Attention(NonlinearSimilarity(encoding_dim))
        self._question_attention_for_question = Attention(NonlinearSimilarity(encoding_dim))
        self._passage_attention_for_answer = Attention(NonlinearSimilarity(encoding_dim), normalize=False)
        self._passage_attention_for_ranking = Attention(NonlinearSimilarity(encoding_dim))

        self._answer_net = nn.GRUCell(encoding_dim, encoding_dim)

        self._v_r_Q = nn.Parameter(torch.rand(encoding_dim)) 
        self._r = r

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializer(self)

    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                passages_length: torch.LongTensor = None,
                correct_passage: torch.LongTensor = None,
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
     
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

        # compute question vector r_Q
        # shape: B x T = attention(B x 2H, B x T x 2H)
        v_r_Q_tiled = self._v_r_Q.unsqueeze(0).expand(batch_size, encoding_dim)
        attn_weights = self._question_attention_for_question(v_r_Q_tiled, encoded_question, question_mask)
        # shape: B x 2H
        r_Q = util.weighted_sum(encoded_question, attn_weights)
        # shape: B x T = attention(B x 2H, B x T x 2H)
        span_start_logits = self._passage_attention_for_answer(r_Q, question_awared_passage, passage_mask)
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)
        span_start_log_probs = util.masked_log_softmax(span_start_logits, passage_mask)
        # shape: B x 2H
        c_t = util.weighted_sum(question_awared_passage, span_start_probs)
        # shape: B x 2H
        h_1 = self._dropout(self._answer_net(c_t, r_Q))

        span_end_logits = self._passage_attention_for_answer(h_1, question_awared_passage, passage_mask)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        span_end_log_probs = util.masked_log_softmax(span_end_logits, passage_mask)

        num_passages = passages_length.size(1)
        acc = Variable(torch.zeros(batch_size, num_passages + 1)).cuda(cuda_device).long()

        acc[:, 1:num_passages+1] = torch.cumsum(passages_length, dim=1)

        g_batch = []
        for b in range(batch_size):
            g = []
            for i in range(num_passages):
                if acc[b, i+1].data[0] > acc[b, i].data[0]:
                    attn_weights = self._passage_attention_for_ranking(r_Q[b:b+1], question_awared_passage[b:b+1, acc[b, i].data[0]: acc[b, i+1].data[0], :], passage_mask[b:b+1, acc[b, i].data[0]: acc[b, i+1].data[0]])
                    r_P = util.weighted_sum(question_awared_passage[b:b+1, acc[b, i].data[0]:acc[b, i+1].data[0], :], attn_weights)
                    question_passage_combined = torch.cat([r_Q[b:b+1], r_P], dim=-1)
                    gi = self._dropout(self._match_layer_2(F.tanh(self._dropout(self._match_layer_1(question_passage_combined)))))
                    g.append(gi)
                else:
                    g.append(Variable(torch.zeros(1, 1)).cuda(cuda_device))
            g = torch.cat(g, dim=1)
            g_batch.append(g)
        
        #t2 = time.time()
        g = torch.cat(g_batch, dim=0)
        passage_log_probs = F.log_softmax(g, dim=-1)

        output_dict = {}
        if span_start is not None:
            AP_loss = F.nll_loss(span_start_log_probs, span_start.squeeze(-1)) +\
                F.nll_loss(span_end_log_probs, span_end.squeeze(-1))
            PR_loss = F.nll_loss(passage_log_probs, correct_passage.squeeze(-1))
            loss = self._r * AP_loss + self._r * PR_loss
            output_dict['loss'] = loss

        _, max_start = torch.max(span_start_probs, dim=1)
        _, max_end = torch.max(span_end_probs, dim=1)
        #t3 = time.time()
        output_dict['span_start_idx'] = max_start
        output_dict['span_end_idx'] = max_end
        #t4 = time.time()
        global ITE
        ITE += 1
        if (ITE % 100 == 0):
            print(" gold %i:%i|predicted %i:%i" %(span_start.squeeze(-1)[0], span_end.squeeze(-1)[0], max_start.data[0], max_end.data[0]))
        
        #t5 = time.time()
        #print("Total: %.5f" % (t5-t0))
        #print("Batch processing 1: %.5f" % (t2-t1))
        #print("Batch processing 2: %.5f" % (t4-t3))
        return output_dict

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

@Model.register("snet-phase2-fix")
class AnswerSynthesis(Model):
    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 question_encoder: Seq2SeqEncoder,
                 passage_encoder: Seq2SeqEncoder,
                 feed_forward: FeedForward,
                 dropout: float = 0.1,
                 num_decoding_steps: int = 40,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AnswerSynthesis, self).__init__(vocab, regularizer)
        self._vocab = vocab
        self._vocab_size = vocab.get_vocab_size() # default: tokens
        self._num_decoding_steps = num_decoding_steps
        self._start_token_index = self._vocab.get_token_index(START_SYMBOL)
        self._end_token_index = self._vocab.get_token_index(END_SYMBOL)

        self._embedder = embedder
        self._question_encoder = question_encoder
        self._passage_encoder = passage_encoder

        encoding_dim = question_encoder.get_output_dim()
        embedding_dim = embedder.get_output_dim()

        self._span_start_embedding = nn.Embedding(2, 50)
        self._span_end_embedding = nn.Embedding(2, 50)
        self._gru_decoder = nn.GRUCell(encoding_dim + embedding_dim, encoding_dim)
        self._feed_forward = feed_forward

        self._attention = Attention(NonlinearSimilarity(encoding_dim))
        
        self._W_r = nn.Linear(embedding_dim, encoding_dim, bias=False)
        self._U_r = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self._V_r = nn.Linear(encoding_dim, encoding_dim, bias=False)

        self._max_out = Maxout(encoding_dim, num_layers=1, output_dims=int(encoding_dim/2), pool_sizes=2)
        self._W_o = nn.Linear(int(encoding_dim/2), self._vocab_size, bias=False)

        self._squad_metrics = SquadEmAndF1()
        #self._predict_acc = CategoricalAccuracy()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        
        initializer(self)
        self._num_iter = 0

    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor,
                span_end: torch.IntTensor,
                answer: Dict[str, torch.LongTensor] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # shape: B x Tq x Eb
        embedded_question = self._embedder(question)
        embedded_passage = self._embedder(passage)

        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)

        # shape: B x Tp
        start_onehot = torch.zeros(batch_size, passage_length)
        start_onehot.scatter_(1, span_start.data, 1)
        end_onehot = torch.zeros(batch_size, passage_length)
        end_onehot.scatter_(1, span_end.data, 1)

        # shape: B x Tp x 50
        embedded_start = self._span_start_embedding(Variable(start_onehot.long(), requires_grad=False))
        embedded_end = self._span_end_embedding(Variable(end_onehot.long(), requires_grad=False))
        embedded_passage = torch.cat([embedded_passage, embedded_start, embedded_end], dim=-1)

        question_mask = util.get_text_field_mask(question)
        passage_mask = util.get_text_field_mask(passage)

        # shape: B x T x 2H
        encoded_question = self._question_encoder(embedded_question, question_mask)
        encoded_passage = self._passage_encoder(embedded_passage, passage_mask)

        # size: 2H
        encoding_dim = encoded_question.size(-1)

        # shape: B x 2H
        last_backward_hidden = torch.cat([encoded_passage[:, 0, int(encoding_dim/2):], encoded_question[:, 0, int(encoding_dim/2):]], dim=-1)
        # shape: B x 2H
        d_t_prev = self._feed_forward(last_backward_hidden)
        
        # shape: B x (Tp + Tq) x 2H
        passage_question_combined = torch.cat([encoded_passage, encoded_question], dim=1)
        passage_question_mask = torch.cat([passage_mask, question_mask], dim=1)

        attn_weights = self._attention(d_t_prev, passage_question_combined, passage_question_mask.float())
        # shape: B x 2H
        c_t_prev = util.weighted_sum(passage_question_combined, attn_weights)

        if answer is not None:
            # shape: B x Ta x Eb = embed(B x Ta)
            embedded_answer = self._embedder(answer)
            # shape: B x Ta
            targets = answer['tokens']
            target_mask = util.get_text_field_mask(answer)
            self._num_decoding_steps = embedded_answer.size(1) - 1

        output_dict = {}

        # TRAINING AND VALIDATING
        if answer is not None:
            # self._num_decoding_steps = embedded_answer.size(1) - 1
            logits = []
            decoded_outputs = []
            for timestep in range(self._num_decoding_steps):
                if self.training:
                    w_t_prev = embedded_answer[:, timestep, :]#.unsqueeze(1)
                else:
                    if timestep == 0:
                        w_t_prev = self._embedder({'tokens': Variable((torch.zeros(batch_size) + self._start_token_index).long())})
                    else:
                        w_t_prev = self._embedder({'tokens' : last_predicted_word})

                # shape: B x (2H + Eb)
                input_combined = torch.cat([c_t_prev, w_t_prev], dim=-1)
                
                d_t = self._gru_decoder(input_combined, d_t_prev)
                
                attn_weights = self._attention(d_t, passage_question_combined)
                c_t = util.weighted_sum(passage_question_combined, attn_weights)
                # shape: B x 2H
                r_t = self._W_r(w_t_prev) + self._U_r(c_t) + self._V_r(d_t)
                # shape: B x H
                m_t = self._max_out(r_t)
                
                # shape: B x V
                word_logits = self._W_o(m_t)
                word_log_probs = F.log_softmax(word_logits, dim=1)
                # shape: B
                _, last_predicted_word = torch.max(word_log_probs, dim=1)
                logits.append(word_logits)

                decoded_outputs.append(last_predicted_word)

                # for next step
                d_t_prev = d_t
                c_t_prev = c_t

            # shape: B x (Ta - 1) x V
            logits = torch.stack(logits, dim=1)
            # loss = F.nll_loss(logits.view(-1, self._vocab_size) , targets[1:].view(-1))
            loss = self._get_loss(logits, targets, target_mask)
            output_dict['loss'] = loss

            output_dict['predicted_answer'] = []
            output_dict['answer_text'] = []
            output_dict['question_text'] = []

            # shape: B x T
            decoded_outputs = torch.stack(decoded_outputs, dim=-1)
            for batch in range(batch_size):
                predicted_answer = self.get_answer_for_validation(decoded_outputs[batch])

                output_dict['predicted_answer'].append(predicted_answer)
                output_dict['question_text'].append(metadata[batch]['question_text'])
                answer_text = metadata[batch]['answer_text']
                output_dict['answer_text'].append(answer_text)

                if answer_text is not None:
                    self._squad_metrics(predicted_answer, answer_text)
                    self._num_iter += 1
                    if (self._num_iter % 50 == 0):
                        print(predicted_answer)

        # INFERENCE
        else:
            beam_size = 12

            d_t_prev_arr = torch.zeros(batch_size, beam_size, encoding_dim)
            c_t_prev_arr = torch.zeros(batch_size, beam_size, encoding_dim)
            d_t_prev_arr[:, 0, :] = d_t_prev
            c_t_prev_arr[:, 0, :] = c_t_prev

            beam_score = torch.zeros(batch_size, beam_size, self._num_decoding_steps)
            beam_word = torch.zeros(batch_size, beam_size, self._num_decoding_steps)
            beam_path = torch.zeros(batch_size, beam_size, self._num_decoding_steps)

            for timestep in range(self._num_decoding_steps):
  
                if timestep == 0:
                    # shape: B x 1 x Eb
                    w_t_prev = self._embedder({'tokens': Variable((torch.zeros(batch_size, 1) + self._start_token_index).long())})
                else:
                    # shape: B x beam_size x Eb
                    w_t_prev = self._embedder({'tokens': Variable(beam_word[:, :, timestep - 1].long())})

                beam_c_t = []
                beam_d_t = []
                beam_predicted_words = []
                beam_predicted_scores = []
                prev_beam_size = w_t_prev.size()[1] 
                for i in range(prev_beam_size):
                    input_combined = torch.cat([c_t_prev_arr[:, i, :], w_t_prev[:, i, :]], dim=-1)
                    
                    # shape: B x 2H = gru(B x 2H, B x 2H)
                    d_t = self._gru_decoder(input_combined, d_t_prev_arr[:, i, :])

                    attn_weights = self._attention(d_t, passage_question_combined)
                    c_t = util.weighted_sum(passage_question_combined, attn_weights)
                    # shape: B x 2H
                    r_t = self._W_r(w_t_prev[:, i, :]) + self._U_r(c_t) + self._V_r(d_t)
                    # shape: B x H
                    m_t = self._max_out(r_t)

                    # shape: B x V
                    word_logits = self._W_o(m_t)
                    word_log_probs = F.log_softmax(word_logits, dim=1)

                    # shape: B x beam_size, predict beam_size words
                    top_predicted_log_probs, top_predicted_words = torch.topk(word_log_probs, beam_size)
                    
                    for batch in range(batch_size):
                        if beam_word[batch, i, timestep - 1] == self._end_token_index:
                            top_predicted_log_probs[batch] = 0 #TODO

                    # shape: B x beam_size = B - B x beam_size
                    if timestep == 0:
                        top_predicted_scores = - top_predicted_log_probs
                    else:
                        top_predicted_scores = beam_score[:, i, timestep - 1].unsqueeze(-1).expand(-1, beam_size) - top_predicted_log_probs

                    beam_c_t.append(c_t)
                    beam_d_t.append(d_t)
                    beam_predicted_words.append(top_predicted_words)
                    beam_predicted_scores.append(top_predicted_scores)

                # shape: B x prev_beam_size x beam_size, stack seq of [prev_beam_size] elements of size [B x beam_size]
                beam_predicted_words = torch.stack(beam_predicted_words, dim=1)
                
                # shape: B x prev_beam_size x beam_size => view => B x (prev_beam_size * beam_size)
                beam_predicted_scores = torch.stack(beam_predicted_scores, dim=1)
                beam_predicted_scores_squeezed = beam_predicted_scores.view(batch_size, -1)

                # shape: B x beam_size
                values, indexs = torch.topk(beam_predicted_scores_squeezed, beam_size, dim=1)

                # tuple(arr_i, arr_j) B x beam_size
                unravel_idxs = np.unravel_index(indexs, (prev_beam_size, beam_size))

                # update d_t_prev_arr
                # sequence of [prev_beam_size] elements of size [B x 2H]
                # => B x prev_beam x 2H
                beam_d_t = torch.stack(beam_d_t, dim=1)
                beam_c_t = torch.stack(beam_c_t, dim=1)

                beam_predicted_words_np = self.var_to_np(beam_predicted_words)
                beam_predicted_scores_np = self.var_to_np(beam_predicted_scores)

                # update beam
                for batch, (i_idxs, j_idxs) in enumerate(zip(unravel_idxs[0], unravel_idxs[1])):
                    beam_word[batch, :, timestep] = beam_predicted_words_np[batch, i_idxs, j_idxs]
                    beam_score[batch, :, timestep] = beam_predicted_scores_np[batch, i_idxs, j_idxs]
                    for idx, i in enumerate(i_idxs):
                        beam_path[batch, idx, timestep] = i
                        d_t_prev_arr[batch, idx] = beam_d_t[batch, i]
                        c_t_prev_arr[batch, idx] = beam_c_t[batch, i]

            # get the best decoded results for each data in batch
            decoded_batch = []
            output_dict['predicted_answer'] = []
            output_dict['question_text'] = []

            for batch in range(batch_size):
                end_token_idxs = np.where(beam_word[batch] == self._end_token_index)
                end_token_scores = beam_score[batch, end_token_idxs[0], end_token_idxs[1]]
                max_idx = np.argmax(end_token_scores)
                beam, step = end_token_idxs[0][max_idx], end_token_idxs[1][max_idx]
                
                decoded_sentence = []
                decoded_sentence.append(beam_word[batch, beam, step])
                for s in range(step, 0, -1):
                    prev_beam = beam_path[batch, beam, s]
                    decoded_sentence.append(beam_word[batch, prev_beam, s - 1])
                decoded_sentence.append(self._start_token_index)
                decoded_sentence.reverse()

                decoded_batch.append(decoded_sentence)

                #TODO:
                evidence = metadata[batch]['evidence']
                predicted_answer = self.get_answer_for_inference(decoded_sentence, evidence)
                
                output_dict['predicted_answer'].append(predicted_answer)
                output_dict['question_text'].append(metadata[batch]['question_text'])
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        #return {'acc': 1, 'em': 1, 'f1': 1}
        em, f1 = self._squad_metrics.get_metric(reset)
        return {
                #'acc': self._predict_acc.get_metric(reset),
                'em': em,
                'f1': f1
                }

    def var_to_np(self, v):
        use_cuda = v.is_cuda
        if use_cuda:
            v = v.data.cpu().numpy()
        else:
            v = v.data.numpy()
        return v

    def get_answer_for_inference(self, decoded_sent: list, evidence: str):
        # if all tokens is unknown return None
        all_unknown = True
        for i in decoded_sent:
            if i != 0:
                all_unknown = False

        if all_unknown:
            return evidence

        vocab = self._vocab.get_index_to_token_vocabulary('tokens')

        # remove duplicate tokens but keep all unknown tokens
        unique_sent = []
        [unique_sent.append(x) for x in decoded_sent if x not in unique_sent or x == 1] # 1 is the index of unknown token

        # if there are still unknown tokens
        unique_sent_np = np.asarray(unique_sent)
        unk = np.where(unique_sent_np == 0)

        tokens = [vocab[t] for t in unique_sent]

        for ix in unk:
            word = self.replace_unknown(vocab, unique_sent, ix, evidence)
            tokens[ix] = word
        
        pred_answer = ' '.join(tokens)

        return pred_answer

    def get_answer_for_validation(self, decoded_output):
        decoded_output_np = self.var_to_np(decoded_output)
        vocab = self._vocab.get_index_to_token_vocabulary('tokens')
        tokens = []
        for word_idx in decoded_output_np:
            if word_idx == self._end_token_index:
                break
            tokens.append(vocab[word_idx])

        pred_answer = ' '.join(tokens)
        return pred_answer

    #TODO:
    def replace_unknown(self):
        pass

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AnswerSynthesis':
        embedder_params = params.pop("text_field_embedder")
        embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        question_encoder = Seq2SeqEncoder.from_params(params.pop("question_encoder"))
        passage_encoder = Seq2SeqEncoder.from_params(params.pop("passage_encoder"))
        feed_forward = FeedForward.from_params(params.pop("feed_forward"))
        dropout = params.pop_float('dropout', 0.1)
        num_decoding_steps = params.pop_int("num_decoding_steps", 40)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   embedder=embedder,
                   question_encoder=question_encoder,
                   passage_encoder=passage_encoder,
                   feed_forward=feed_forward,
                   dropout=dropout,
                   num_decoding_steps=num_decoding_steps,
                   initializer=initializer,
                   regularizer=regularizer)
