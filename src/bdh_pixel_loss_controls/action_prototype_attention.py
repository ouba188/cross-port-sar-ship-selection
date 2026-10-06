"""Action prototypes and port-conditioned retrieval; no novelty guarantee."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from report_bdh_pixel_handoff import require


def make_bank(features, labels, base, expert, ports, pseudocount=8.):
    """Aggregate already-purged source rows, shrinking toward shared action means."""
    features = np.asarray(features, dtype=np.float32)
    labels, base, expert, ports = map(np.asarray, (labels, base, expert, ports))
    require(len(features) == len(labels) == len(base) == len(expert) == len(ports), 'bank row mismatch')
    require(pseudocount > 0 and len(features) > 0, 'nonempty bank and positive shrinkage required')
    names, port_ids = np.unique(ports, return_inverse=True)
    action = 64 * labels + 8 * base + expert
    group = 512 * port_ids + action
    shared = {int(a): features[action == a].mean(0) for a in np.unique(action)}
    records, vectors = [], []
    for g in np.unique(group):
        rows = np.flatnonzero(group == g)
        first, count = rows[0], len(rows)
        weight = count / (count + pseudocount)
        vectors.append(weight * features[rows].mean(0) + (1 - weight) * shared[int(action[first])])
        records.append((labels[first], base[first], expert[first], port_ids[first], count))
    y, b, e, p, n = np.asarray(records, dtype=np.int64).T
    events = (b == y).astype(int) + 2 * (e == y).astype(int)
    evidence = np.concatenate((np.eye(8)[y], np.eye(4)[events]), axis=1).astype(np.float32)
    return dict(features=np.stack(vectors), labels=y, base=b, expert=e, ports=p,
                port_names=names, counts=n, evidence=evidence)


def bank_tensors(bank, device):
    return {key: torch.as_tensor(value, device=device) for key, value in bank.items() if key != 'port_names'}


class ActionPrototypeAttention(nn.Module):
    """Same parameters, either pooled prototype or hierarchical port retrieval."""

    def __init__(self, feature_dim, width=32, mode='pooled'):
        super().__init__()
        if mode not in ('pooled', 'hierarchical'):
            raise ValueError('unknown retrieval mode')
        self.mode = mode
        self.project = nn.Sequential(nn.Linear(feature_dim, width), nn.LayerNorm(width))
        self.reliability = nn.Sequential(nn.Linear(2 * width + 12, width), nn.Tanh(),
                                         nn.Linear(width, 1, bias=False))

    def forward(self, features, bank):
        query = F.normalize(self.project(features), dim=-1)
        if not len(bank['features']):
            probability = query.sum(-1, keepdim=True) * 0 + features.new_full((len(features), 8), 1 / 8)
            return dict(probabilities=probability, port_weights=features.new_zeros((len(features), 0)))
        keys = F.normalize(self.project(bank['features']), dim=-1)
        evidence = bank['evidence']
        similarity = 8 * (query @ keys.T)
        ports = torch.unique(bank['ports'], sorted=True)
        if self.mode == 'pooled':
            joined = torch.cat((query[:, None, :].expand(-1, len(keys), -1),
                                keys[None].expand(len(query), -1, -1),
                                evidence[None].expand(len(query), -1, -1)), dim=-1)
            weights = (similarity + self.reliability(joined).squeeze(-1)).softmax(-1)
            probability = weights @ evidence[:, :8]
            port_weights = torch.stack([weights[:, bank['ports'] == port].sum(-1) for port in ports], dim=-1)
        else:
            contexts, port_scores = [], []
            for port in ports:
                use = bank['ports'] == port
                weights = similarity[:, use].softmax(-1)
                context_key, context_evidence = weights @ keys[use], weights @ evidence[use]
                contexts.append(context_evidence[:, :8])
                port_scores.append(self.reliability(torch.cat((query, context_key, context_evidence), dim=-1)))
            port_weights = torch.cat(port_scores, dim=-1).softmax(-1)
            probability = (torch.stack(contexts, dim=1) * port_weights[:, :, None]).sum(1)
        probability = probability + 1e-4
        return dict(probabilities=probability / probability.sum(-1, keepdim=True), port_weights=port_weights)


def source_objective(probabilities, labels, base, expert, row_weights):
    """Class-balanced source supervision plus explicit repair/harm event loss."""
    rows = torch.arange(len(labels), device=labels.device)
    pb, pe = probabilities[rows, base], probabilities[rows, expert]
    agree = base == expert
    zero = torch.zeros_like(pb)
    events = torch.stack((torch.where(agree, 1 - pb, 1 - pb - pe),
                          torch.where(agree, zero, pb), torch.where(agree, zero, pe),
                          torch.where(agree, pb, zero)), dim=1)
    targets = base.eq(labels).long() + 2 * expert.eq(labels).long()
    class_loss = F.nll_loss(probabilities.clamp_min(1e-8).log(), labels, reduction='none')
    event_loss = F.nll_loss(events.clamp_min(1e-8).log(), targets, reduction='none')
    return ((class_loss + event_loss) * row_weights).mean()


def transport_rows(conditional, base, expert, density):
    """Redistribute each source class/pair mass over its unlabeled query objects."""
    conditional, density = np.asarray(conditional, dtype=float), np.asarray(density, dtype=float)
    base, expert = np.asarray(base), np.asarray(expert)
    require(density.shape == (len(base), 8) and np.isfinite(density).all() and (density > 0).all(),
            'positive finite class densities required')
    require(conditional.shape == (8, 8, 8) and np.isfinite(conditional).all()
            and (conditional >= 0).all(), 'invalid source class/pair mass')
    pair = 8 * base + expert
    mass = np.empty_like(density)
    for group in np.unique(pair):
        use = pair == group
        weights = density[use] / density[use].sum(0, keepdims=True)
        mass[use] = weights * conditional[:, group // 8, group % 8]
    return mass


def robust_row_scores(mass, base, expert, observed):
    """Shared support family, pointwise lower scores; labels are not query inputs."""
    base, expert, observed = np.asarray(base), np.asarray(expert), np.asarray(observed)
    require(observed.shape == (8,) and observed.dtype == bool and observed.any(), 'invalid observed support')
    supports = ((np.arange(1, 256)[:, None] >> np.arange(8)) & 1).astype(bool)
    supports = supports[np.all(supports[:, observed], axis=1)]
    rows = np.arange(len(base))
    values = (supports[:, expert] * mass[rows, expert] - supports[:, base] * mass[rows, base])
    return len(base) * (values / supports.sum(1, keepdims=True)).min(0)
