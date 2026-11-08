"""Constructive probabilistic network baseline for traffic event classification."""
import copy

import numpy as np
from scipy.special import logsumexp


class CPNN:
    def __init__(self, seed, dimension, cap=4096):
        self.cap, self.dimension = cap, dimension
        self.centers = np.empty((cap+1, dimension), dtype=np.float64)
        self.labels = np.empty(cap+1, dtype=np.int64)
        self.radii = np.empty(cap+1, dtype=np.float64)
        self.counts = np.zeros(cap+1, dtype=np.int64)
        self.ids = np.empty(cap+1, dtype=np.int64)
        self.n, self.next_id, self.stable, self.pass_number = 0, 0, 0, 0
        self.rng = np.random.Generator(np.random.PCG64(seed))

    @staticmethod
    def radius(distance):
        return np.clip(distance/np.sqrt(-np.log(.1)), 1e-6, 10.)

    def state(self):
        return dict(cap=self.cap, dimension=self.dimension, n=self.n, next_id=self.next_id,
            stable=self.stable, pass_number=self.pass_number, rng=copy.deepcopy(self.rng.bit_generator.state),
            centers=self.centers[:self.n].copy(), labels=self.labels[:self.n].copy(),
            radii=self.radii[:self.n].copy(), counts=self.counts[:self.n].copy(), ids=self.ids[:self.n].copy())

    def load(self, state):
        assert self.cap == state['cap'] and self.dimension == state['dimension']
        for key in ['n', 'next_id', 'stable', 'pass_number']:
            setattr(self, key, state[key])
        for key in ['centers', 'labels', 'radii', 'counts', 'ids']:
            getattr(self, key)[:self.n] = state[key]
        self.rng.bit_generator.state = state['rng']

    def observe(self, x, label):
        n = self.n
        delta = self.centers[:n]-x
        distances = np.sqrt(np.einsum('ij,ij->i', delta, delta)/self.dimension)
        foreign = self.labels[:n] != label
        same = ~foreign
        cover = same & (distances <= self.radii[:n]*np.sqrt(-np.log(.4)))
        added = deleted = cap_hit = 0
        if np.any(cover):
            eligible = np.flatnonzero(cover)
            order = np.lexsort((self.ids[eligible], distances[eligible]))
            self.counts[eligible[order[0]]] += 1
        else:
            dmin = float(distances[foreign].min()) if np.any(foreign) else None
            self.centers[n] = x
            self.labels[n] = label
            self.radii[n] = self.radius(dmin) if dmin is not None else 10.
            self.counts[n] = 1
            self.ids[n] = self.next_id
            self.next_id += 1
            self.n += 1
            added = 1
        old = self.radii[:n].copy()
        new = np.minimum(old[foreign], self.radius(distances[foreign]))
        assert np.all(new <= old[foreign])
        self.radii[:n][foreign] = new
        relative = float(np.max(np.abs(self.radii[:n]-old)/np.maximum(old, 1e-6))) if n else 0.
        if self.n > self.cap:
            cap_hit = deleted = 1
            frequencies = np.bincount(self.labels[:self.n], minlength=3)
            eligible = np.flatnonzero(frequencies[self.labels[:self.n]] > 1)
            order = np.lexsort((self.ids[eligible], self.counts[eligible]))
            victim = eligible[order[0]]
            for key in ['centers', 'labels', 'radii', 'counts', 'ids']:
                a = getattr(self, key)
                a[victim] = a[self.n-1]
            self.n -= 1
        return added, deleted, cap_hit, relative

    def epoch(self, x, y, epoch):
        self.counts[:self.n] = 0
        totals = np.zeros(3, dtype=np.int64)
        initial_radii = dict(zip(self.ids[:self.n].tolist(), self.radii[:self.n].tolist()))
        for row in self.rng.permutation(len(x)):
            a, d, h, relative = self.observe(np.asarray(x[row], dtype=np.float64).ravel(), int(y[row]))
            totals += (a, d, h)
        max_change = max((abs(self.radii[i]-initial_radii[int(self.ids[i])])/initial_radii[int(self.ids[i])]
                          for i in range(self.n) if int(self.ids[i]) in initial_radii), default=0.)
        self.stable = self.stable+1 if totals[0] == 0 and max_change < 1e-6 else 0
        self.pass_number = epoch
        assert self.n <= self.cap and np.all((self.radii[:self.n] >= 1e-6) & (self.radii[:self.n] <= 10))
        return dict(phase='construction', train_loss='', lr='', optimizer_steps_net1='', optimizer_steps_net2='',
                    prototype_count=self.n, new_prototypes=int(totals[0]), deleted_prototypes=int(totals[1]),
                    cap_trigger_count=int(totals[2]), max_relative_radius_change=max_change,
                    stable_passes=self.stable)

    def predict(self, x, batch=128):
        out = np.empty((len(x), 3), dtype=np.float64)
        centers = self.centers[:self.n]
        # Compute in float64 log space, retaining finite values for tiny radii.
        for start in range(0, len(x), batch):
            xx = np.asarray(x[start:start+batch], dtype=np.float64).reshape(-1, self.dimension)
            distance2 = np.zeros((len(xx), self.n))
            for channel in range(self.dimension):
                distance2 += (xx[:, channel, None]-centers[None, :, channel])**2
            logkernel = -distance2/self.dimension/self.radii[:self.n][None, :]**2
            weights = np.full(self.n, -np.inf)
            np.log(self.counts[:self.n], out=weights, where=self.counts[:self.n] > 0)
            # Remove the common (possibly enormous) negative offset before
            # adding log-counts or aggregating classes. This preserves count
            # ratios and avoids cancellation when radii reach their floor.
            logkernel -= logkernel[:, self.counts[:self.n] > 0].max(axis=1, keepdims=True)
            scores = np.stack([logsumexp(logkernel[:, self.labels[:self.n] == k] + weights[self.labels[:self.n] == k], axis=1)
                               for k in range(3)], axis=1)
            scores -= scores.max(axis=1, keepdims=True)
            out[start:start+len(xx)] = np.exp(scores-logsumexp(scores, axis=1, keepdims=True))
        assert np.isfinite(out).all()
        return out.astype(np.float32)
