# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
from typing import Tuple, Union, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import k2
from k2.rnnt_loss import _adjust_pruning_lower_bound
from cif import BATPredictor
from encoder_interface import EncoderInterface
from label_smoothing import LabelSmoothingLoss
from scaling import penalize_abs_values_gt

from icefall.utils import make_pad_mask


class BoundaryAwareTransducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: nn.Module,
        predictor: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        vocab_size: int,
        ignore_index: int = -1,
        transducer_weight: float = 1.0,
        predictor_weight: float = 1.0,
        cif_weight: float = 1.0,
        r_d: int = 3,
        r_u: int = 5,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.joiner = joiner

        self.simple_am_proj = nn.Linear(
            encoder_dim,
            vocab_size,
        )
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)
        self.ignore_index = ignore_index

        self.criterion_quantity = nn.L1Loss()

        self.cif_weight = cif_weight
        # if self.cif_weight > 0:
        self.cif_output_layer = nn.Linear(encoder_dim, vocab_size)
        self.criterion_ce = LabelSmoothingLoss(ignore_index=self.ignore_index)

        self.transducer_weight = transducer_weight
        self.predictor_weight = predictor_weight

        self.r_d = r_d
        self.r_u = r_u

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        assert x.size(0) == x_lens.size(0) == len(y)

        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        # Encoder out mask for predictor
        encoder_out_mask = (~make_pad_mask(encoder_out_lens)[:, None, :]).to(encoder_out.device)
        
        decoder_out = self.forward_decoder(y)
        
        y_lens = torch.IntTensor([len(tokens) for tokens in y]).to(encoder_out.device)
        y_padded_ignore = pad_sequence(y, batch_first=True, padding_value=float(self.ignore_index))

        # Now for the predictor
        (
            pre_acoustic_embeds,
            pre_token_length,
            pre_peak_index,
        ) = self.forward_predictor(encoder_out, y_padded_ignore, encoder_out_mask)

        loss_cif_qua = self.criterion_quantity(
            y_lens.type_as(pre_token_length), pre_token_length
        )

        # if self.cif_weight > 0.0:
        cif_predict = self.cif_output_layer(pre_acoustic_embeds)
        loss_cif_ce = self.criterion_ce(cif_predict, y_padded_ignore)
        # else:
        #     loss_cif_ce = 0.0

        # Note: y does not start with SOS
        # y_padded : [B, S]
        # y_padded_blank = y.pad(mode="constant", padding_value=0)

        # y_padded_blank = y_padded_blank.to(torch.int64)
        y_padded_blank = pad_sequence(y, batch_first=True, padding_value=0)
        y_padded_blank = y_padded_blank.to(device=encoder_out.device, dtype=torch.int64)
        boundary = torch.zeros((encoder_out.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens.float()
        boundary[:, 3] = encoder_out_lens.float()

        pre_peak_index = torch.floor(pre_peak_index).long()
        # print(pre_peak_index[0])
        s_begin = pre_peak_index - self.r_d

        T = encoder_out.size(1)
        B = encoder_out.size(0)

        mask = torch.arange(0, T, device=encoder_out.device).reshape(1, T).expand(B, T)
        mask = mask <= boundary[:, 3].reshape(B, 1) - 1

        s_begin_padding = boundary[:, 2].reshape(B, 1) - (self.r_u + self.r_d) + 1
        # handle the cases where `len(symbols) < s_range`
        s_begin_padding = torch.clamp(s_begin_padding, min=0)

        s_begin = torch.where(mask, s_begin, s_begin_padding)

        mask2 = s_begin < boundary[:, 2].reshape(B, 1) - (self.r_u + self.r_d) + 1

        s_begin = torch.where(
            mask2, s_begin, boundary[:, 2].reshape(B, 1) - (self.r_u + self.r_d) + 1
        )

        s_begin = torch.clamp(s_begin, min=0)
        
        # s_begin = _adjust_pruning_lower_bound(s_begin, self.r_d + self.r_u)
        
        ranges = s_begin.reshape((B, T, 1)).expand(
            (B, T, min(self.r_u + self.r_d, min(y_lens)))
        ) + torch.arange(
            min(self.r_d + self.r_u, min(y_lens)), device=encoder_out.device
        )

        # Test
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens
        
        am = self.simple_am_proj(encoder_out)
        lm = self.simple_lm_proj(decoder_out)
        
        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded_blank,
                termination_symbol=self.decoder.blank_id,
                lm_only_scale=0.0,
                am_only_scale=0.0,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=6,
        )
        # print(ranges[0, :, :])
        # print(ranges.shape)
        # exit(1)
        # Test end
        
        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded_blank,
                ranges=ranges,
                termination_symbol=self.decoder.blank_id,
                boundary=boundary,
                reduction="sum",
            )

        bat_loss = self.transducer_weight * pruned_loss
        quantity_loss = self.predictor_weight * loss_cif_qua
        ce_loss = self.cif_weight * loss_cif_ce

        # if torch.isinf(pruned_loss):
        #     print(s_begin[1])
        #     print(min(self.r_u + self.r_d, min(y_lens)))
        #     print(ranges[1,:,:])
        #     am = self.simple_am_proj(encoder_out)
        #     lm = self.simple_lm_proj(decoder_out)
            
        #     with torch.cuda.amp.autocast(enabled=False):
        #         simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
        #             lm=lm.float(),
        #             am=am.float(),
        #             symbols=y_padded_blank,
        #             termination_symbol=self.decoder.blank_id,
        #             lm_only_scale=0.0,
        #             am_only_scale=0.0,
        #             boundary=boundary,
        #             reduction="sum",
        #             return_grad=True,
        #         )

        #     # ranges : [B, T, prune_range]
        #     ranges = k2.get_rnnt_prune_ranges(
        #         px_grad=px_grad,
        #         py_grad=py_grad,
        #         boundary=boundary,
        #         s_range=6,
        #     )
        #     am_pruned, lm_pruned = k2.do_rnnt_pruning(
        #         am=self.joiner.encoder_proj(encoder_out),
        #         lm=self.joiner.decoder_proj(decoder_out),
        #         ranges=ranges,
        #         )

        # # logits : [B, T, prune_range, vocab_size]

        # # project_input=False since we applied the decoder's input projections
        # # prior to do_rnnt_pruning (this is an optimization for speed).
        #     logits = self.joiner(am_pruned, lm_pruned, project_input=False)
        #     print(ranges[1, :, :])
        #     with torch.cuda.amp.autocast(enabled=False):
        #         pruned_loss = k2.rnnt_loss_pruned(
        #             logits=logits.float(),
        #             symbols=y_padded_blank,
        #             ranges=ranges,
        #             termination_symbol=self.decoder.blank_id,
        #             boundary=boundary,
        #             reduction="sum",
        #         )
        #     print(pruned_loss)
        #     exit(1)
        return bat_loss, quantity_loss, ce_loss

    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs

        Args:
            x (torch.Tensor): A 3-D tensor of shape (N, T, C)
            x_lens (torch.Tensor): A 1-D tensor of shape (N,)

        Returns:
            encoder_out: A 3-D tensor of shape (N, T, C)
            encoder_out_lens: A 1-D tensor of shape (N,)
        """
        x, x_lens = self.encoder_embed(x, x_lens)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_decoder(self, y: List[torch.Tensor]) -> torch.Tensor:
        """compute decoder

        Args:
            y (k2.RaggedTensor):
                A ragged tensor with 2 axes [utt][label]. It contains labels of each
                utterance
            y_lens (torch.Tensor):
                A tensor with then lengths of y

        Returns:
            torch.Tensor:
                Decoder out
        """

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1]. start with SOS.
        # sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = pad_sequence(sos_y, batch_first=True, padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        return decoder_out

    def forward_predictor(
        self,
        encoder_out: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        encoder_out_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(
            encoder_out,
            y,
            encoder_out_mask,
            ignore_id=self.ignore_index,
        )
        return pre_acoustic_embeds, pre_token_length, pre_peak_index
    
def add_sos(token_ids: List[torch.Tensor], sos_id: int) -> List[torch.Tensor]:
    return [torch.cat((torch.Tensor([sos_id]).to(tokens.device), tokens), 0) for tokens in token_ids]