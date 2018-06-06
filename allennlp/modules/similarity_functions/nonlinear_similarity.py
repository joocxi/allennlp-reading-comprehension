from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction

class NonlinearSimilarity(SimilarityFunction):
    def __init__(self,
                 encoding_dim: int) -> None:
        super(NonlinearSimilarity, self).__init__()
        self._linear_1 = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self._linear_2 = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self._similarity = nn.Linear(encoding_dim, 1, bias=False)

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        return self._similarity(F.tanh(self._linear_1(tensor_1) + self._linear_2(tensor_2))).squeeze(-1)
