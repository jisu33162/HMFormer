"""All gradient methods; trainer sees only the condition-specific labels."""
import copy
import math

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.nn import functional as F

from .common import RandomStreams, seed_all, rng_state, restore_rng
from .data import loader
from .baseline_models import Transformer, CNN
from . import config


def cpu_order(values):
    return np.argsort(values.detach().cpu().numpy(), kind='stable')


def remember(batch, rho, epoch):
    return max(1, math.floor(batch*(1-rho*min((epoch-1)/9, 1))))


def choose_split(prob, batch):
    n = len(prob)
    assert n >= 2*batch and np.isfinite(prob).all()
    mask = prob > .5
    fallback = int(mask.sum() < batch or (~mask).sum() < batch)
    if fallback:
        count = int(np.clip(mask.sum(), batch, n-batch))
        order = np.lexsort((np.arange(n), -prob))
        mask = np.zeros(n, dtype=bool)
        mask[order[:count]] = True
    return np.flatnonzero(mask), np.flatnonzero(~mask), fallback


class Trainer:
    def __init__(self, task, length, device, batch, *, training=None, total_epochs=100,
                 final_lr=5e-6, warmup_epochs=10):
        self.task, self.device, self.batch = task, torch.device(device), batch
        self.training = dict(config.TRAINING if training is None else training)
        self.method, self.seed = task['method'], task['train_seed']
        self.streams = RandomStreams(self.seed)
        dual = self.method in ['Co-teaching', 'JoCoR', 'DivideMix']
        self.models = []
        for i in range(2 if dual else 1):
            seed_all(self.seed+i)
            if self.method == 'CNN-TID-adapt':
                model = CNN(length, dropout=self.training['dropout'])
            else:
                model = Transformer(length, width=self.training['width'], heads=self.training['heads'],
                                    layers=self.training['layers'], dropout=self.training['dropout'],
                                    feedforward_dim=self.training.get('feedforward_dim'))
            self.models.append(model.to(self.device))
        seed_all(self.seed+3000001)
        groups = [list(m.parameters()) for m in self.models]
        self.optimizers = ([self.optimizer(groups[0]+groups[1], self.training)] if self.method == 'JoCoR'
                           else [self.optimizer(g, self.training) for g in groups])
        # Explicit scheduler state is an epoch counter plus a fully specified function.
        self.scheduler = dict(last_epoch=0, warmup_epochs=warmup_epochs, total_epochs=total_epochs,
                              start=self.training['lr'], end=final_lr)
        self.steps = [0]*len(self.models)
        self.last_audit = {}

    @staticmethod
    def optimizer(params, training=None):
        settings = config.TRAINING if training is None else training
        return torch.optim.AdamW(params, lr=settings['lr'], weight_decay=settings['weight_decay'],
                                 betas=settings.get('betas', (.9, .999)), eps=settings.get('eps', 1e-8))

    def state(self):
        return self.snapshot()

    def load(self, state):
        for m, s in zip(self.models, state['models']):
            m.load_state_dict(s)
            m.to(self.device)
        for o, s in zip(self.optimizers, state['optimizers']):
            o.load_state_dict(s)
            # Adam's non-capturable scalar step stays on CPU; moments follow weights.
            for group in o.state.values():
                for key, val in group.items():
                    if torch.is_tensor(val) and key != 'step':
                        group[key] = val.to(self.device)
        self.scheduler = state['scheduler'].copy()
        self.steps = state['steps'].copy()
        self.streams.restore(state['streams'])
        restore_rng(state['rng'])

    def snapshot(self):
        # Do not move live parameters: optimizer references and states stay intact.
        return dict(models=[{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in self.models],
                    optimizers=copy.deepcopy([o.state_dict() for o in self.optimizers]),
                    scheduler=self.scheduler.copy(), steps=self.steps.copy(),
                    streams=self.streams.state(), rng=rng_state())

    def update(self, loss, opt, model_indices):
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for i in model_indices for p in self.models[i].parameters()]
        torch.nn.utils.clip_grad_norm_(parameters, self.training['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        for i in model_indices:
            self.steps[i] += 1

    def epoch(self, x, y, epoch):
        warmup = self.scheduler['warmup_epochs']
        start, end = self.scheduler['start'], self.scheduler['end']
        duration = max(1, self.scheduler['total_epochs'] - warmup)
        lr = start if epoch <= warmup else end + .5*(start-end)*(1+math.cos(math.pi*(epoch-warmup)/duration))
        for opt in self.optimizers:
            for group in opt.param_groups:
                group['lr'] = lr
        before = self.steps.copy()
        self.last_audit = {}
        if self.method == 'DivideMix':
            result = self.divide_epoch(x, y, epoch)
        else:
            result = self.supervised_epoch(x, y, epoch)
        self.scheduler['last_epoch'] = epoch
        result.update(lr=lr, optimizer_steps_net1=self.steps[0]-before[0],
                      optimizer_steps_net2=self.steps[1]-before[1] if len(self.steps) == 2 else '')
        return result

    def supervised_epoch(self, x, y, epoch):
        for m in self.models:
            m.train()
        sums, seen, min_kept = {}, 0, self.batch
        # Exactly one shared batch iterator for Co-teaching and JoCoR.
        for xx, yy, rows in loader(x, y, self.batch, self.streams.torch['shuffle']):
            xx, yy = xx.to(self.device), yy.to(self.device)
            logits = [m(xx) for m in self.models]
            ce = [F.cross_entropy(z, yy, reduction='none') for z in logits]
            keep = remember(len(yy), self.task['noise_rate'], epoch)
            min_kept = min(min_kept, keep)
            if self.method == 'Co-teaching':
                idx = [torch.as_tensor(cpu_order(v)[:keep], device=self.device) for v in ce]
                losses = [ce[0][idx[1]].mean(), ce[1][idx[0]].mean()]
                self.update(losses[0], self.optimizers[0], [0])
                self.update(losses[1], self.optimizers[1], [1])
                parts = dict(train_loss=sum(v.item() for v in losses)/2, ce_net1=losses[0].item(), ce_net2=losses[1].item())
            elif self.method == 'JoCoR':
                logp = [F.log_softmax(z, dim=1) for z in logits]
                p = [z.exp() for z in logp]
                # Both KL directions retain gradients through both distributions.
                kl = (p[0]*(logp[0]-logp[1])).sum(1)+(p[1]*(logp[1]-logp[0])).sum(1)
                joint = .1*(ce[0]+ce[1]) + .9*kl
                ids = torch.as_tensor(cpu_order(joint)[:keep], device=self.device)
                loss = joint[ids].mean()
                self.update(loss, self.optimizers[0], [0, 1])
                parts = dict(train_loss=loss.item(), ce_net1=ce[0][ids].mean().item(),
                             ce_net2=ce[1][ids].mean().item(), consistency_kl=kl[ids].mean().item())
            else:
                loss = ce[0].mean()
                self.update(loss, self.optimizers[0], [0])
                parts = dict(train_loss=loss.item(), ce_net1=loss.item())
            seen += len(yy)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.) + v*len(yy)
            if self.method in ['Co-teaching', 'JoCoR']:
                assert len(logits[0]) == len(rows) == len(logits[1])
                if self.task['noise_rate'] == 0:
                    assert keep == len(rows)
        self.last_audit = dict(shared_batch_iterator=True, seen=seen, min_kept=min_kept)
        return dict(phase='supervised', **{k: v/seen for k, v in sums.items()})

    @torch.no_grad()
    def loss_vector(self, model, x, y, generator=None):
        model.eval()
        losses = np.full(len(x), np.nan, dtype=np.float64)
        counts = np.zeros(len(x), dtype=np.int64)
        for xx, yy, rows in loader(x, y, self.batch, generator):
            values = F.cross_entropy(model(xx.to(self.device)), yy.to(self.device), reduction='none').cpu().numpy()
            losses[rows.numpy()] = values
            counts[rows.numpy()] += 1
        assert (counts == 1).all() and np.isfinite(losses).all()
        return losses

    def divide_epoch(self, x, y, epoch):
        if epoch <= self.scheduler['warmup_epochs']:
            totals = []
            for i, m in enumerate(self.models):
                m.train()
                total, seen = 0., 0
                for xx, yy, rows in loader(x, y, self.batch, self.streams.torch[f'warm{i+1}']):
                    loss = F.cross_entropy(m(xx.to(self.device)), yy.to(self.device))
                    self.update(loss, self.optimizers[i], [i])
                    total += loss.item()*len(yy)
                    seen += len(yy)
                totals.append(total/seen)
            return dict(phase='warmup', train_loss=float(np.mean(totals)), ce_net1=totals[0], ce_net2=totals[1])
        probabilities, gmm_info = [], []
        # Both posteriors are fixed before either network receives any update.
        for i, m in enumerate(self.models):
            loss = self.loss_vector(m, x, y)
            normalized = (loss-loss.min())/max(float(loss.max()-loss.min()), 1e-8)
            gmm = GaussianMixture(2, max_iter=100, tol=.01, reg_covar=5e-4, n_init=1,
                                  random_state=(self.seed+1000*epoch+i+1) % 2**32)
            gmm.fit(normalized[:, None])
            prob = gmm.predict_proba(normalized[:, None])[:, gmm.means_.argmin()]
            assert np.isfinite(prob).all()
            probabilities.append(prob)
            gmm_info.append(dict(converged=bool(gmm.converged_), iterations=int(gmm.n_iter_)))
        self.last_audit = dict(probabilities=[p.copy() for p in probabilities], gmm=gmm_info)
        totals, record = [], dict(phase='semi_supervised')
        for i in range(2):
            prob = probabilities[1-i]
            labeled, unlabeled, fallback = choose_split(prob, self.batch)
            parts = self.mixmatch(i, x, y, labeled, unlabeled, prob, epoch)
            totals.append(parts)
            record.update({f'labeled_count_net{i+1}': len(labeled), f'unlabeled_count_net{i+1}': len(unlabeled),
                           f'gmm_converged_net{i+1}': gmm_info[i]['converged'],
                           f'gmm_iterations_net{i+1}': gmm_info[i]['iterations'],
                           f'split_fallback_count_net{i+1}': fallback})
        for key in totals[0]:
            record[key] = float(np.mean([v[key] for v in totals]))
        return record

    def augment(self, x):
        noise = torch.randn(x.shape, generator=self.streams.torch['augmentation'], dtype=torch.float32)
        return (x + .01*noise).to(self.device)

    @staticmethod
    def sharpen(p):
        p = p.pow(2)
        return p/p.sum(1, keepdim=True).clamp_min(1e-8)

    def mixmatch(self, i, x, y, labeled, unlabeled, prob, epoch):
        net, peer = self.models[i], self.models[1-i]
        peer_versions = [p._version for p in peer.parameters()]
        net.train()
        peer.eval()
        ll = loader(x, y, self.batch, self.streams.torch[f'labeled{i+1}'], labeled)
        ul = loader(x, y, self.batch, self.streams.torch[f'unlabeled{i+1}'], unlabeled)
        ui = iter(ul)
        sums, seen = {}, 0
        for batch_index, (xx, yy, rows) in enumerate(ll):
            assert np.array_equal(yy.numpy(), np.asarray(y)[rows.numpy()])
            try:
                uu, _, urows = next(ui)
            except StopIteration:
                ui = iter(ul)  # Each new cycle draws a new permutation.
                uu, _, urows = next(ui)
            x1, x2, u1, u2 = self.augment(xx), self.augment(xx), self.augment(uu), self.augment(uu)
            with torch.no_grad():
                weight = torch.from_numpy(prob[rows.numpy()].astype(np.float32)).to(self.device)[:, None]
                current = F.one_hot(yy.to(self.device), 3).float()
                px = (net(x1).softmax(1) + net(x2).softmax(1))/2
                tx = self.sharpen(weight*current + (1-weight)*px)
                pu = (net(u1).softmax(1)+net(u2).softmax(1)+peer(u1).softmax(1)+peer(u2).softmax(1))/4
                tu = self.sharpen(pu)
            assert not tx.requires_grad and not tu.requires_grad
            inputs = torch.cat([x1, x2, u1, u2])
            targets = torch.cat([tx, tx, tu, tu])
            a = self.streams.numpy.beta(4, 4)
            a = max(a, 1-a)
            permutation = torch.randperm(len(inputs), generator=self.streams.torch['mix_permutation']).to(self.device)
            mixed = a*inputs + (1-a)*inputs[permutation]
            target = a*targets + (1-a)*targets[permutation]
            logits = net(mixed)
            boundary = 2*len(xx)
            lx = -(F.log_softmax(logits[:boundary], 1)*target[:boundary]).sum(1).mean()
            lu = (logits[boundary:].softmax(1)-target[boundary:]).square().mean()
            meanp = logits.softmax(1).mean(0).clamp_min(1e-8)
            prior = (torch.log((1/3)/meanp)/3).sum()
            # Start at weight 0 after warmup, then ramp up over 16 semi-supervised epochs.
            first_epoch = self.scheduler['warmup_epochs'] + 1
            coefficient = 25*float(np.clip((epoch-first_epoch + batch_index/len(ll))/16, 0, 1))
            loss = lx + coefficient*lu + prior
            self.update(loss, self.optimizers[i], [i])
            parts = dict(train_loss=loss.item(), soft_ce=lx.item(), unsupervised_mse=lu.item(),
                         prior_kl=prior.item(), unsupervised_weight=coefficient)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.)+len(xx)*v
            seen += len(xx)
        assert peer_versions == [p._version for p in peer.parameters()], 'Peer was updated in the wrong training phase'
        return {k: v/seen for k, v in sums.items()}

    @torch.no_grad()
    def predict(self, x):
        for m in self.models:
            m.eval()
        predictions = np.empty((len(x), 3), dtype=np.float32)
        for xx, _, rows in loader(x, None, self.batch):
            xx = xx.to(self.device)
            prob = sum(m(xx).softmax(1) for m in self.models)/len(self.models)
            predictions[rows.numpy()] = prob.cpu().numpy()
        assert np.isfinite(predictions).all()
        return predictions
