from typing import Tuple
from overrides import overrides
from allennlp.training.metrics.metric import Metric

import pdb

@Metric.register("msmarco")
class Rouge(Metric):
    def __init__(self) -> None:
        self._total_rouge = 0.0
        self._count = 0
        self._beta = 1.2

    @overrides
    def __call__(self, best_span_string, answer_strings):
        prec = []
        rec = []
        predicted_tokens = best_span_string.split(" ")
        for answer_string in answer_strings:
            answer_tokens = answer_string.split(" ")
            lcs = self.my_lcs(answer_tokens, predicted_tokens)
            prec.append(lcs/float(len(predicted_tokens)))
            rec.append(lcs/float(len(answer_tokens)))

        prec_max = max(prec)
        rec_max = max(rec)

        if (prec_max != 0 and rec_max != 0):
            score = ((1 + self._beta**2)*prec_max*rec_max)/float(rec_max + self._beta**2*prec_max)
        else:
            score = 0.0

        self._total_rouge += score
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        rouge_l = self._total_rouge / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return rouge_l

    @overrides
    def reset(self):
        self._total_rouge = 0.0
        self._count = 0

    def __str__(self):
        return f"Rouge(rouge={self._total_rouge})"

    def my_lcs(self, string, sub):
        """
        Calculates longest common subsequence for a pair of tokenized strings
        :param string : list of str : tokens from a string split using whitespace
        :param sub : list of str : shorter string, also split using whitespace
        :returns: length (list of int): length of the longest common subsequence between the two strings
        Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
        """
        if(len(string)< len(sub)):
            sub, string = string, sub

        lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

        for j in range(1,len(sub)+1):
            for i in range(1,len(string)+1):
                if(string[i-1] == sub[j-1]):
                    lengths[i][j] = lengths[i-1][j-1] + 1
                else:
                    lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

        return lengths[len(string)][len(sub)]
