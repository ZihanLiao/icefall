from typing import Optional, Tuple, Union

import k2

import numpy as np
import torch
from torch import nn


def cif(
    hidden: torch.Tensor, alphas: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, len_time, hidden_size = hidden.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = (
            torch.ones([batch_size], device=hidden.device) - integrate
        )

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - torch.ones([batch_size], device=hidden.device),
            integrate,
        )
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(
            fire_place[:, None].repeat(1, hidden_size),
            remainds[:, None] * hidden[:, t, :],
            frame,
        )

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        l = torch.index_select(
            frames[b, :, :], 0, torch.nonzero(fire >= threshold).squeeze()
        )
        pad_l = torch.zeros(
            [max_label_len - l.size(0), hidden_size], device=hidden.device
        )
        list_ls.append(torch.cat([l, pad_l], 0))
    return torch.stack(list_ls, 0), fires


def cif_wo_hidden(alphas: torch.Tensor, threshold: float) -> torch.Tensor:
    batch_size, len_time = alphas.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=alphas.device)
    # intermediate vars along time
    list_fires = []

    for t in range(len_time):
        alpha = alphas[:, t]

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - torch.ones([batch_size], device=alphas.device) * threshold,
            integrate,
        )

    fires = torch.stack(list_fires, 1)
    return fires


class CifPredictor(nn.Module):
    """
    CifPredictor

    Modified from https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/models/predictor/cif.py#L509
    """

    def __init__(
        self,
        idim: int,
        l_order: int,
        r_order: int,
        threshold: float = 1.0,
        dropout: float = 0.1,
        smooth_factor: float = 1.0,
        smooth_factor2: float = 1.0,
        noise_threshold: float = 0.0,
        noise_threshold2: float = 0.0,
        upsample_times: int = 5,
        upsample_type: str = "cnn",
        tail_threshold: float = 0.0,
        tail_mask: bool = True,
        use_cif1_cnn: bool = True,
    ) -> None:
        super().__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = nn.Linear(idim, 1)
        self.threshold = threshold
        self.dropout = nn.Dropout(p=dropout)
        self.smooth_factor = smooth_factor
        self.smooth_factor2 = smooth_factor2
        self.noise_threshold = noise_threshold
        self.noise_threshold2 = noise_threshold2
        self.tail_threshold = tail_threshold
        self.tail_mask = tail_mask

        self.upsample_times = upsample_times
        self.upsample_type = upsample_type

        assert self.upsample_type in (
            "cnn",
            "cnn_blstm",
            "cnn_attn",
        ), f"{self.upsample_type} did not implemented"

        if self.upsample_type == "cnn":
            self.upsample_cnn = nn.ConvTranspose1d(
                idim, idim, self.upsample_times, self.upsample_times
            )
            self.cif_output2 = nn.Linear(idim, 1)
        elif self.upsample_Type == "cnn_blstm":
            self.upsample_cnn = nn.ConvTranspose1d(
                idim, idim, self.upsample_times, self.upsample_times
            )
            self.blstm = nn.LSTM(
                idim,
                idim,
                1,
                bias=True,
                batch_first=True,
                dropout=0.0,
                bidirectional=True,
            )
            self.cif_output2 = nn.Linear(idim * 2, 1)

        self.use_cif1_cnn = use_cif1_cnn

    def forward(
        self,
        encoder_out: torch.Tensor,
        target_label: Optional[Union[torch.Tensor, k2.RaggedTensor]] = None,
        target_label_length: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        ignore_id: int = -1,
        mask_chunk_predictor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        h = encoder_out
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))

        if not self.use_cif1_cnn:
            _output = context
        else:
            _output = output

        if self.upsample_type == "cnn":
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1, 2)
        elif self.upsample_type == "cnn_blstm":
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1, 2)
            output2, (_, _) = self.blstm(output2)

        alphas2 = torch.sigmoid(self.cif_output2(output2))
        alphas2 = nn.functional.relu(
            alphas2 * self.smooth_factor2 - self.noise_threshold2
        )

        if mask is not None:
            mask2 = (
                mask.repeat(1, self.upsample_times, 1)
                .transpose(-1, -2)
                .reshape(alphas2.shape[0], -1)
            )
            mask2 = mask2.unsqueeze(-1)
            alphas2 = alphas2 * mask2
        alphas2 = alphas2.squeeze(-1)
        token_num2 = alphas2.sum(-1)

        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mask = mask.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(
                encoder_out, alphas, token_num, mask=mask
            )

        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak, token_num2

    def get_upsample_timestamp(
        self,
        hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        token_num: Optional[int] = None,
    ):
        h = hidden
        b = hidden.shape[0]
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))

        # alphas2 is an extra head for timestamp prediction
        if not self.use_cif1_cnn:
            _output = context
        else:
            _output = output
        if self.upsample_type == "cnn":
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1, 2)
        elif self.upsample_type == "cnn_blstm":
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1, 2)
            output2, (_, _) = self.blstm(output2)
        elif self.upsample_type == "cnn_attn":
            output2 = self.upsample_cnn(_output)
            output2 = output2.transpose(1, 2)
            output2, _ = self.self_attn(output2, mask)
        alphas2 = torch.sigmoid(self.cif_output2(output2))
        alphas2 = torch.nn.functional.relu(
            alphas2 * self.smooth_factor2 - self.noise_threshold2
        )
        # repeat the mask in T demension to match the upsampled length
        if mask is not None:
            mask2 = (
                mask.repeat(1, self.upsample_times, 1)
                .transpose(-1, -2)
                .reshape(alphas2.shape[0], -1)
            )
            mask2 = mask2.unsqueeze(-1)
            alphas2 = alphas2 * mask2
        alphas2 = alphas2.squeeze(-1)
        _token_num = alphas2.sum(-1)
        if token_num is not None:
            alphas2 *= (token_num / _token_num)[:, None].repeat(1, alphas2.size(1))
        # re-downsample
        ds_alphas = alphas2.reshape(b, -1, self.upsample_times).sum(-1)
        ds_cif_peak = cif_wo_hidden(ds_alphas, self.threshold - 1e-4)
        # upsampled alphas and cif_peak
        us_alphas = alphas2
        us_cif_peak = cif_wo_hidden(us_alphas, self.threshold - 1e-4)
        return ds_alphas, ds_cif_peak, us_alphas, us_cif_peak

    def tail_process_fn(
        self,
        hidden: torch.Tensor,
        alphas: torch.Tensor,
        token_num: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(
                alphas.device
            )
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor


class BATPredictor(nn.Module):
    def __init__(
        self,
        idim: int,
        l_order: int,
        r_order: int,
        threshold: float = 1.0,
        dropout: float = 0.1,
        smooth_factor: float = 1.0,
        noise_threshold: float = 0,
        return_accum: bool = False,
    ) -> None:
        super().__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order+r_order+1, groups=idim)
        self.cif_output = nn.Linear(idim, 1)
        self.threshold = threshold
        self.dropout = nn.Dropout(p=dropout)
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.return_accum = return_accum

    def forward(
        self,
        hidden: torch.Tensor,
        target_label: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        ignore_id: int = -1,
        mask_chunk_predictor: Optional[torch.Tensor] = None,
        target_label_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        h = hidden
        N, T, C = h.shape
        context = h.transpose(1, 2)  # (N, T, C) -> (N, C, T)
        queries = self.pad(context)  # (N, C, T) -> (N, C, T + l_order + r_order)
        memory = self.cif_conv1d(queries)
        output = memory + context
        output = self.dropout(output)
        output = torch.relu(output.transpose(1, 2))
        output = self.cif_output(output)
        alphas = torch.sigmoid(output) # (N, T, 1)
        alphas = nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        # (N, T)
        if mask is not None:
            alphas = alphas * mask.transpose(-1, -2).float()
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor

        alphas = alphas.squeeze(-1)
        if target_label_length:
            target_length = target_label_length
        elif target_label is not None:
            # target_length = target_label.shape.row_splits(1)[1:] - target_label.shape.row_splits(1)[:-1]
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= ((target_length + 1e-4) / token_num)[:, None].repeat(
                1, alphas.size(1)
            )
        acoustic_embeds, fires = cif(
            hidden, alphas, self.threshold
        )
        cif_peak = fires.cumsum(-1)
        # acoustic_embeds, cif_peak = self.cif(
        #     hidden, alphas, self.threshold, self.return_accum,
        # )
        token_num = alphas.sum(-1)
        token_num_int = torch.max(token_num).type(torch.int32).item()
        acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak

    def cif(
        self,
        input: torch.Tensor,
        alpha: torch.Tensor,
        beta: float = 1.0,
        return_accum: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, C = input.size()
        assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"

        dtype = alpha.dtype
        alpha = alpha.float()
        alpha_sum = alpha.sum(1)
        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

        # aggregate and integrate
        csum = alpha.cumsum(-1)
        with torch.no_grad():
            # indices used for scattering
            right_idx = (csum / beta).floor().long().clamp(max=T)
            left_idx = right_idx.roll(1, dims=1)
            left_idx[:, 0] = 0

            # count # of fires from each source
            fire_num = right_idx - left_idx
            extra_weights = (fire_num - 1).clamp(min=0)
            # The extra entry in last dim is for
            output = input.new_zeros((B, T + 1, C))
            source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
            zero = alpha.new_zeros((1,))
 
        # right scatter
        fire_mask = fire_num > 0
        right_weight = torch.where(
            fire_mask, csum - right_idx.type_as(alpha) * beta, zero
        ).type_as(input)
        assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
        output.scatter_add_(
            1,
            right_idx.unsqueeze(-1).expand(-1, -1, C),
            right_weight.unsqueeze(-1) * input,
        )

        # left scatter
        left_weight = (
            alpha - right_weight - extra_weights.type_as(alpha) * beta
        ).type_as(input)
        output.scatter_add_(
            1,
            left_idx.unsqueeze(-1).expand(-1, -1, C),
            left_weight.unsqueeze(-1) * input,
        )

        # extra scatters
        if extra_weights.ge(0).any():
            extra_steps = extra_weights.max().item()
            tgt_idx = left_idx
            src_feats = input * beta
            for _ in range(extra_steps):
                tgt_idx = tgt_idx + 1
                tgt_idx = tgt_idx.clamp(max=T)
                # tgt_idx = (tgt_idx + 1).clamp(max=T)
                # (B, S, 1)
                src_mask = extra_weights > 0
                output.scatter_add_(
                    1,
                    tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                    src_feats * src_mask.unsqueeze(2),
                )
                extra_weights -= 1
        output = output[:, :T, :]

        if return_accum:
            return output, csum
        else:
            return output, alpha
