"""Random streams used by the baseline trainers."""
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def rng_state():
    out = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.backends.mps.is_available():
        out['mps'] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        out['cuda'] = torch.cuda.get_rng_state_all()
    return out


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


class RandomStreams:
    """Every named independent stream is recorded and checkpointed."""
    names = ['shuffle', 'warm1', 'warm2', 'labeled1', 'unlabeled1', 'labeled2',
             'unlabeled2', 'augmentation', 'mix_permutation']

    def __init__(self, seed):
        self.seeds = {n: (seed + 100003*(i+1)) % 2**32 for i, n in enumerate(self.names)}
        # The ordinary/common training shuffle uses the training seed itself.
        self.seeds['shuffle'] = seed
        self.torch = {n: torch.Generator(device='cpu').manual_seed(s) for n, s in self.seeds.items()}
        self.seeds['mix_beta'] = (seed + 2000003) % 2**32
        self.numpy = np.random.Generator(np.random.PCG64(self.seeds['mix_beta']))

    def state(self):
        return dict(torch={n: g.get_state() for n, g in self.torch.items()},
                    numpy=self.numpy.bit_generator.state, seeds=self.seeds)

    def restore(self, s):
        assert self.seeds == s['seeds']
        for n, st in s['torch'].items():
            self.torch[n].set_state(st)
        self.numpy.bit_generator.state = s['numpy']
