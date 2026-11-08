"""Seven ablation variants with one attention implementation."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from . import backbone as BACKBONE
from . import hmm as BASE

FEATURE = {'V2', 'V4', 'V5', 'V6', 'V7'}
RELATION = {'V3', 'V4', 'V5', 'V6', 'V7'}


class AblationModel(nn.Module):
    def __init__(self, variant, steps, initialization, hmm_state=None, *,
                 width=128, heads=8, layers=3, dropout=.1, states=6,
                 gate_strength=.5, additive_strength=.5, feedforward_dim=None,
                 covariance_floor=.001, covariance_type='full'):
        super().__init__()
        assert variant in {f'V{i}' for i in range(1, 8)}
        self.variant = variant
        self.heads = heads
        self.gate_strength = gate_strength
        self.additive_strength = additive_strength
        self.backbone = BACKBONE.TinyTransformer(steps, width=width, heads=heads, layers=layers,
                                               dropout=dropout, feedforward_dim=feedforward_dim)
        self.backbone.load_state_dict(initialization['backbone'])
        if variant != 'V1':
            self.hmm = BASE.SharedARHMM(states=states, ar_order=0 if variant == 'V6' else 1,
                                      covariance_floor=covariance_floor, covariance_type=covariance_type)
            if hmm_state is not None:
                self.hmm.load_state_dict(hmm_state)
            self.hmm.requires_grad_(False)
            self.hmm.eval()
        if variant in FEATURE:
            local_dim = states + states * states
            self.qkv_hmm = nn.ModuleList([nn.Linear(local_dim, 3 * width, bias=False) for _ in range(layers)])
            self.qkv_hmm.load_state_dict(initialization['qkv_hmm'])
        if variant in RELATION:
            self.relations = nn.ModuleList([BASE.TemporalRelationBias(heads, 1, states, 1, steps)
                                           for _ in range(layers)])
            self.relations.load_state_dict(initialization['relations'])

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'hmm'):
            self.hmm.eval()
        return self

    def fuse(self, content, relation):
        if self.variant == 'V4':
            return content + self.additive_strength * torch.tanh(relation)
        return content * (1 + self.gate_strength * torch.tanh(relation))

    def forward(self, x, *, hmm_input=None, diagnostic=None):
        local = pairs = None
        if hasattr(self, 'hmm'):
            state_input = x if hmm_input is None else hmm_input
            assert state_input.shape == x.shape
            with torch.no_grad():
                posterior = self.hmm(state_input)
                if self.variant in FEATURE:
                    local = BASE.temporal_features(posterior)
                if self.variant in RELATION:
                    pairs = BASE.ordered_pairs(posterior)
        else:
            assert hmm_input is None
        h = self.backbone.input(x) + self.backbone.position
        batch, steps, width = h.shape
        for index, layer in enumerate(self.backbone.encoder.layers):
            normalized = layer.norm1(h)
            qkv = F.linear(normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            if self.variant in FEATURE:
                qkv = qkv + self.qkv_hmm[index](local)
            q, k, v = [part.reshape(batch, steps, self.heads, width // self.heads).transpose(1, 2)
                       for part in qkv.chunk(3, dim=-1)]
            content = (q @ k.transpose(-1, -2)) / math.sqrt(width // self.heads)
            scores = content
            if self.variant in RELATION:
                relation = self.relations[index](pairs)
                scores = self.fuse(content, relation)
                if diagnostic is not None:
                    diagnostic.update(index, content, relation)
            weights = F.dropout(scores.softmax(-1), p=layer.self_attn.dropout, training=self.training)
            attended = (weights @ v).transpose(1, 2).reshape(batch, steps, width)
            h = h + layer.dropout1(layer.self_attn.out_proj(attended))
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(h)))))
            h = h + layer.dropout2(feedforward)
        return self.backbone.output(self.backbone.norm(h.mean(1)))
