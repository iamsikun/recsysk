"""AUGRU (GRU with attentional update gate) from DIEN.

Reference
---------
Zhou, Guorui, et al. "Deep interest evolution network for click-through
rate prediction." AAAI 2019. Eq. 2-5 (GRU) and Eq. 11-12 (AUGRU).

The vanilla GRU update is::

    u_t = sigmoid(W_u i_t + U_u h_{t-1} + b_u)
    r_t = sigmoid(W_r i_t + U_r h_{t-1} + b_r)
    h~_t = tanh(W_h i_t + r_t * (U_h h_{t-1}) + b_h)
    h_t  = (1 - u_t) * h_{t-1} + u_t * h~_t

AUGRU scales the *vector* update gate element-wise by a scalar attention
score a_t (one per time step)::

    u~_t = a_t * u_t
    h_t  = (1 - u~_t) * h_{t-1} + u~_t * h~_t

This preserves the per-dimension information in u_t while letting the
attention score dim-the whole update for time steps whose interests are
less relevant to the target item.
"""

from __future__ import annotations

import torch
from torch import nn


class AUGRUCell(nn.Module):
    """A single AUGRU step (DIEN Eq. 11-12)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Stack the three (u, r, h~) linear projections for efficiency.
        self.x_to_uhr = nn.Linear(input_size, 3 * hidden_size)
        self.h_to_uhr = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        att_score: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the AUGRU state by one step.

        Args:
            x: (B, input_size) — current-step input.
            h_prev: (B, hidden_size) — previous hidden state.
            att_score: (B,) — attention score for this step (0..1 after
                softmax across time). Applied as a scalar scale on the
                update gate.

        Returns:
            (B, hidden_size) — new hidden state.
        """
        gates_x = self.x_to_uhr(x)
        gates_h = self.h_to_uhr(h_prev)

        x_u, x_r, x_h = gates_x.chunk(3, dim=-1)
        h_u, h_r, h_h = gates_h.chunk(3, dim=-1)

        u = torch.sigmoid(x_u + h_u)
        r = torch.sigmoid(x_r + h_r)
        h_tilde = torch.tanh(x_h + r * h_h)

        # Attentional update gate: scale u element-wise by a_t.
        u_att = att_score.unsqueeze(-1) * u
        h_new = (1.0 - u_att) * h_prev + u_att * h_tilde
        return h_new


class DynamicAUGRU(nn.Module):
    """Unrolls :class:`AUGRUCell` over a padded sequence with a mask.

    At each step, valid positions advance the cell; padded positions
    (mask == False) carry the previous hidden state forward unchanged.
    The input layout matches the rest of the codebase: batch-first,
    ``(B, T, input_size)`` with a boolean ``(B, T)`` mask where True means
    "valid behavior".
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cell = AUGRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
        att_scores: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run AUGRU over the full sequence.

        Args:
            inputs: (B, T, input_size) — per-step inputs (typically the
                interest-extractor GRU's hidden states).
            att_scores: (B, T) — per-step scalar attention scores.
            mask: (B, T) boolean — True for valid positions. If ``None``,
                all positions are considered valid.

        Returns:
            (B, hidden_size) — final hidden state after the last valid
            step. For rows with no valid positions, the zero vector.
        """
        batch_size, seq_len, _ = inputs.shape
        h = inputs.new_zeros(batch_size, self.hidden_size)

        if mask is None:
            for t in range(seq_len):
                h = self.cell(inputs[:, t, :], h, att_scores[:, t])
            return h

        for t in range(seq_len):
            h_new = self.cell(inputs[:, t, :], h, att_scores[:, t])
            keep = mask[:, t].unsqueeze(-1)
            h = torch.where(keep, h_new, h)
        return h
