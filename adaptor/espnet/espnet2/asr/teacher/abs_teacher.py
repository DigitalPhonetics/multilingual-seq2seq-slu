from abc import ABC
from abc import abstractmethod

from typing import Optional
import torch


class AbsTeacher(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: Optional[torch.Tensor],
        speech: Optional[torch.Tensor],
        speech_lengths: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError
