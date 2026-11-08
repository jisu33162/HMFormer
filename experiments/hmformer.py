"""Posterior features in Q/K/V and multiplicative attention gating."""
import math
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import TinyTransformer
from .hmm import (ARHMM, SharedARHMM, TemporalRelationBias, fuse_attention_scores,
                  ordered_pairs, temporal_features)


class TinyTransformerHMMGate(nn.Module):
    """Keep the TinyTransformer backbone; add HMM features to Q/K/V and gate QK."""

    def __init__(self, steps=24, width=128, heads=8, layers=3, dropout=0.1,
                 hmm_state=None, hmm_states=6, gate_strength=.5,
                 hmm_ar_order=1, hmm_covariance_floor=1e-3, hmm_covariance_type="full",
                 hmm_evidence_scale=None, train_hmm=False, norm_order="pre",
                 hmm_relation_grad=False, relation_fusion="multiplicative", hmm_models=1,
                 feedforward_dim=None):
        super().__init__()
        if not isinstance(hmm_states, int) or hmm_states < 2:
            raise ValueError("hmm_states must be an integer >= 2")
        if type(hmm_models) is not int or hmm_models < 1:
            raise ValueError("hmm_models must be a positive integer")
        self.hmm_models = hmm_models
        self.gate_strength = float(gate_strength)
        if norm_order not in ("pre", "post"):
            raise ValueError("norm_order must be pre or post")
        self.norm_order = norm_order
        if relation_fusion not in ("multiplicative", "additive", "content_only"):
            raise ValueError("relation_fusion must be multiplicative, additive or content_only")
        if relation_fusion != "multiplicative" and self.gate_strength != .5:
            raise ValueError("gate_strength applies only to multiplicative fusion")
        self.relation_fusion = relation_fusion
        if not math.isfinite(self.gate_strength) or not 0 < self.gate_strength < 1:
            raise ValueError("gate_strength must be between 0 and 1, excluding both endpoints")
        self.hmm_evidence_scale = None if hmm_evidence_scale is None else float(hmm_evidence_scale)
        if (self.hmm_evidence_scale is not None
                and (not math.isfinite(self.hmm_evidence_scale) or self.hmm_evidence_scale <= 0)):
            raise ValueError("hmm_evidence_scale must be finite and positive, or None")
        self.backbone = TinyTransformer(steps, width=width, heads=heads,
                                        layers=layers, dropout=dropout,
                                        feedforward_dim=feedforward_dim)
        hmm_options = dict(states=hmm_states, ar_order=hmm_ar_order,
                           covariance_floor=hmm_covariance_floor,
                           covariance_type=hmm_covariance_type)
        # Preserve the singleton construction and its RNG consumption exactly.
        self.hmm = (SharedARHMM(**hmm_options) if hmm_models == 1
                    else ARHMM(models=hmm_models, **hmm_options))
        if hmm_state is not None:
            self.hmm.load_state_dict(hmm_state)
        self.hmm.requires_grad_(False)
        self.hmm.eval()
        if type(train_hmm) is not bool:
            raise ValueError("train_hmm must be bool")
        self.train_hmm = train_hmm
        if type(hmm_relation_grad) is not bool:
            raise ValueError("hmm_relation_grad must be bool")
        if hmm_relation_grad and not train_hmm:
            raise ValueError("hmm_relation_grad requires train_hmm=True")
        if hmm_relation_grad and relation_fusion == "content_only":
            raise ValueError("content_only has no relation branch for HMM gradients")
        self.hmm_relation_grad = hmm_relation_grad
        if train_hmm:
            self.hmm.requires_grad_(True)
        self.heads = heads
        local_dim = hmm_models * (hmm_states + hmm_states * hmm_states + int(self.hmm_evidence_scale is not None))
        self.qkv_hmm = nn.ModuleList([
            nn.Linear(local_dim, 3 * width, bias=False)
            for _ in range(layers)
        ])
        self.relations = nn.ModuleList([
            TemporalRelationBias(heads, hmm_models, hmm_states, 1, steps)
            for _ in range(layers)
        ])
        for projection in self.qkv_hmm:
            nn.init.zeros_(projection.weight)
        if self.relation_fusion == "content_only":
            self.relations.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.hmm.eval()
        return self

    def fuse_scores(self, content, relation):
        if self.relation_fusion == "additive":
            return fuse_attention_scores(content, relation, "attention")
        if self.relation_fusion == "content_only":
            return content
        if self.gate_strength == .5:
            return fuse_attention_scores(content, relation, "attention_gate")
        return content * (1.0 + self.gate_strength * torch.tanh(relation))

    def embed_inputs(self, x, local):
        """Default: one token and one local HMM feature vector per time point."""
        return self.backbone.input(x) + self.backbone.position, local

    def forward_features(self, x, *, force_identity_gate=False, hmm_input=None,
                         return_local_features=False, return_hmm_likelihood=False,
                         hmm_likelihood_interval=None):
        if type(return_local_features) is not bool:
            raise ValueError("return_local_features must be bool")
        if type(return_hmm_likelihood) is not bool:
            raise ValueError("return_hmm_likelihood must be bool")
        if return_hmm_likelihood and not return_local_features:
            raise ValueError("HMM likelihood output requires return_local_features=True")
        if hmm_likelihood_interval is not None:
            if (not isinstance(hmm_likelihood_interval, (list, tuple))
                    or len(hmm_likelihood_interval) != 2
                    or any(type(i) is not int for i in hmm_likelihood_interval)
                    or not 0 <= hmm_likelihood_interval[0] < hmm_likelihood_interval[1] <= x.shape[1]):
                raise ValueError("HMM likelihood interval requires 0 <= start < stop <= steps")
            if not return_hmm_likelihood:
                raise ValueError("HMM likelihood interval requires likelihood output")
        # Self-supervised masking can hide observations from the input encoder
        # while inferring HMM states from the original training sequence.
        # The regular classification path leaves hmm_input unset.
        state_input = x if hmm_input is None else hmm_input
        if state_input.shape != x.shape:
            raise ValueError("hmm_input must match the input sequence shape")
        with (nullcontext() if self.train_hmm and self.training else torch.no_grad()):
            posterior = (self.hmm(state_input) if self.hmm_evidence_scale is None and hmm_likelihood_interval is None else
                         self.hmm(state_input, return_evidence=True))
            local = temporal_features(posterior)
            if self.hmm_evidence_scale is not None:
                evidence = posterior["log_evidence"].permute(0, 2, 1)
                local = torch.cat((local, torch.asinh(evidence / self.hmm_evidence_scale)), dim=-1)
            # Default retains the historical local-only HMM gradient path.
            # The optional full path changes training derivatives only;
            # all-position posterior values and inference stay identical.
            with (nullcontext() if self.hmm_relation_grad and self.train_hmm and self.training
                  else torch.no_grad()):
                pairs = ordered_pairs(posterior)
        h, local = self.embed_inputs(x, local)
        batch, steps = h.shape[:2]
        width_per_head = h.shape[-1] // self.heads
        for layer, projection, relation in zip(self.backbone.encoder.layers,
                                               self.qkv_hmm, self.relations):
            normalized = layer.norm1(h) if self.norm_order == "pre" else h
            qkv = F.linear(normalized, layer.self_attn.in_proj_weight,
                           layer.self_attn.in_proj_bias) + projection(local)
            q, k, v = [part.reshape(batch, steps, self.heads, width_per_head)
                       .transpose(1, 2) for part in qkv.chunk(3, dim=-1)]
            content = (q @ k.transpose(-1, -2)) / math.sqrt(width_per_head)
            scores = (content if force_identity_gate or self.relation_fusion == "content_only" else
                      self.fuse_scores(content, relation(pairs)))
            weights = F.dropout(scores.softmax(dim=-1), p=layer.self_attn.dropout,
                                training=self.training)
            attended = (weights @ v).transpose(1, 2).reshape(batch, steps, -1)
            h = h + layer.dropout1(layer.self_attn.out_proj(attended))
            if self.norm_order == "post":
                h = layer.norm1(h)
            feedforward = layer.linear2(layer.dropout(layer.activation(
                layer.linear1(layer.norm2(h) if self.norm_order == "pre" else h))))
            h = h + layer.dropout2(feedforward)
            if self.norm_order == "post":
                h = layer.norm2(h)
        if return_hmm_likelihood:
            # Reuse the likelihood from this exact posterior computation.
            likelihood = posterior["log_likelihood"]
            if hmm_likelihood_interval is not None:
                start, stop = hmm_likelihood_interval
                likelihood = posterior["log_evidence"][..., start:stop].sum(dim=-1)
            return h, local, likelihood
        return (h, local) if return_local_features else h

    def forward(self, x, *, force_identity_gate=False):
        h = self.forward_features(x, force_identity_gate=force_identity_gate)
        return self.backbone.output(self.backbone.norm(h.mean(dim=1)))
