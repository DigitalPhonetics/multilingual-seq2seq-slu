#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Teacher."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict, Any

import torch
from espnet2.asr.decoder.hugging_face_transformers_decoder import (
    HuggingFaceTransformersDecoder,  # noqa: H301
)
from espnet2.asr.teacher.abs_teacher import AbsTeacher
from espnet2.asr.teacher.loss.square_alignment import (
    EyedMaskedNormalizedSquareAlignmentL2Loss,
    MaskedNormalizedSquareAlignmentL2Loss,
    NormalizedSquareAlignmentL1Loss,
    NormalizedSquareAlignmentL2Loss,
    SquareAlignmentL1Loss,
    SquareAlignmentL2Loss,
)
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.train.class_choices import ClassChoices
from espnet2.utils.sized_dict import SizedDict
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from typeguard import check_argument_types

try:
    from transformers import AutoModel, AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        l1=torch.nn.L1Loss,
        square_alignment_l1=SquareAlignmentL1Loss,
        square_alignment_l2=SquareAlignmentL2Loss,
        normalized_square_alignment_l1=NormalizedSquareAlignmentL1Loss,
        normalized_square_alignment_l2=NormalizedSquareAlignmentL2Loss,
        masked_normalized_square_alignment_l2=MaskedNormalizedSquareAlignmentL2Loss,
        eyed_masked_normalized_square_alignment_l2=EyedMaskedNormalizedSquareAlignmentL2Loss,
    ),
    type_check=torch.nn.Module,
    default="l1",
    optional=False,
)


class HuggingFaceTransformersTeacher(AbsTeacher):
    """Hugging Face Transformers Teacher."""

    def __init__(
        self,
        model_name_or_path: str,
        loss: str,
        token_type: str,
        token_list: Union[Path, str, Iterable[str]],
        bpemodel: Union[Path, str, Iterable[str]],
        average_output: bool = False,
        post_mapper: bool = False,
        post_encoder: bool = True,
        post_decoder: bool = False,
        loss_conf: Dict[str, Any] = {},
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.post_mapper = post_mapper
        self.post_encoder = post_encoder
        self.post_decoder = post_decoder

        model = AutoModel.from_pretrained(model_name_or_path)

        if self.post_mapper:
            if model.config.is_encoder_decoder:
                encoder = model.encoder
            else:
                encoder = model

            self.teacher_embed_tokens = encoder.embed_tokens
            self.teacher_embed_positions = encoder.embed_positions
            self.teacher_embed_scale = encoder.embed_scale

            for m in [self.teacher_embed_tokens, self.teacher_embed_positions]:
                for p in m.parameters():
                    p.requires_grad = False

        if self.post_encoder or self.post_decoder:
            if model.config.is_encoder_decoder:
                self.teacher_encoder = model.encoder
            else:
                self.teacher_encoder = model

            for param in self.teacher_encoder.parameters():
                param.requires_grad = False

        if self.post_decoder:
            # HuggingFaceTransformersDecoder class is used for
            # the inputs' processing between Encoder and Decoder
            self.teacher_decoder = HuggingFaceTransformersDecoder(
                0,  # vocab_size is ignored in this class
                model.decoder.config.hidden_size,
                model_name_or_path,
                return_hidden=True,
            )

            for param in self.teacher_decoder.parameters():
                param.requires_grad = False

        self.teacher_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if token_type == "bpe" or token_type == "hugging_face":
            self.tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        else:
            self.tokenizer = build_tokenizer(token_type=token_type)

        self.converter = TokenIDConverter(token_list=token_list)
        self.average_output = average_output

        loss_class = loss_choices.get_class(loss)
        self.loss = loss_class(**loss_conf)

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: Optional[torch.Tensor],
        decoder_out_lens: Optional[torch.Tensor],
        mapper_out: Optional[torch.Tensor],
        mapper_out_lens: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        text_in: Optional[torch.Tensor],
        text_in_lengths: Optional[torch.Tensor],
        lids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Calculate the Teacher output and Student output's distance from it."""

        if self.tokenizer.model == self.teacher_tokenizer.name_or_path:
            sentences = self.teacher_tokenizer.batch_decode(
                text, skip_special_tokens=True
            )
        else:
            sentences = []
            for i in range(text.shape[0]):
                token_int = text[i][: text_lengths[i]].tolist()
                token = self.converter.ids2tokens(token_int)
                sentences.append(self.tokenizer.tokens2text(token))

        # Tokenize sentences
        encoded_input = self.teacher_tokenizer(
            sentences, padding=True, return_tensors="pt"
        )

        if lids is not None:
            encoded_input["input_ids"][:, 0:1] = lids

        encoded_input["attention_mask"] = encoded_input["attention_mask"].to(
            encoder_out.device
        )
        encoded_input["input_ids"] = encoded_input["input_ids"].to(encoder_out.device)

        encoded_input["return_dict"] = True

        loss_list = []

        if self.post_mapper:
            with torch.no_grad():
                input_embeds = (
                    self.teacher_embed_tokens(encoded_input["input_ids"])
                    * self.teacher_embed_scale
                )
                pos_embeds = self.teacher_embed_positions(
                    encoded_input["input_ids"].shape
                )
                teacher_encoder_input = input_embeds + pos_embeds

            loss = self.loss(
                mapper_out, teacher_encoder_input, encoded_input["attention_mask"], mapper_out_lens
            )
            loss_list.append(loss)

        if self.post_encoder or self.post_decoder:
            # Compute token embeddings
            with torch.no_grad():
                teacher_encoder_output_dict = self.teacher_encoder(**encoded_input)
                teacher_encoder_output = teacher_encoder_output_dict.last_hidden_state

            if self.post_encoder:
                if self.average_output:
                    # Perform pooling
                    teacher_encoder_output = mean_pooling(
                        teacher_encoder_output, encoded_input["attention_mask"]
                    )
                    # Normalize embeddings
                    teacher_encoder_output = torch.nn.functional.normalize(
                        teacher_encoder_output, p=2, dim=1
                    )
                    teacher_encoder_output = teacher_encoder_output.unsqueeze(1)

                    loss = self.loss(encoder_out, teacher_encoder_output)
                else:
                    loss = self.loss(
                        encoder_out,
                        teacher_encoder_output,
                        encoded_input["attention_mask"],
                        encoder_out_lens,
                    )

                loss_list.append(loss)

            if self.post_decoder:
                with torch.no_grad():
                    teacher_decoder_output, _ = self.teacher_decoder(
                        teacher_encoder_output,
                        encoded_input["attention_mask"].sum(1),
                        text_in,
                        text_in_lengths,
                    )

                decoder_out_mask = (
                    (~make_pad_mask(decoder_out_lens)).int().to(decoder_out.device)
                )

                loss = self.loss(
                    decoder_out,
                    teacher_decoder_output,
                    decoder_out_mask,
                    decoder_out_lens,
                )

                loss_list.append(loss)

        return torch.stack(loss_list).mean()


# From https://huggingface.co/sentence-transformers/all-mpnet-base-v2
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    )
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
