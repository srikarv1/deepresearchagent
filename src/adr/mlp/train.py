"""One fixed CPU MLP experiment and untuned linear diagnostics on frozen inputs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, precision_recall_fscore_support, roc_auc_score)
import sklearn
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from adr.mlp.prepare import FEATURE_NAMES, LABELS, TEXT_FIELDS, load_prepared


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prepared: Path
    output_root: Path
    seed: int = 42
    input_dimensions: int = 4623
    hidden_dimensions: tuple[int, int] = (256, 64)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    threshold: float = 0.5
    logistic_C: float = 1.0
    logistic_max_iter: int = 2000


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def build_model(cfg: TrainingConfig) -> nn.Sequential:
    first, second = cfg.hidden_dimensions
    return nn.Sequential(nn.Linear(cfg.input_dimensions, first), nn.ReLU(), nn.Dropout(cfg.dropout),
                         nn.Linear(first, second), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(second, 1))


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float = .5) -> dict:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probabilities')
    predictions = (p >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=[0, 1], zero_division=0)
    return {'threshold': threshold, 'accuracy': float(accuracy_score(y, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
            'SKIP': {'precision':float(precision[0]), 'recall':float(recall[0]),
                     'f1':float(f1[0]), 'support':int(support[0])},
            'EXECUTE': {'precision':float(precision[1]), 'recall':float(recall[1]),
                        'f1':float(f1[1]), 'support':int(support[1])},
            'macro_f1': float(f1.mean()),
            'confusion_matrix':confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
            'confusion_order':'rows=true, columns=predicted; [SKIP, EXECUTE]',
            'roc_auc':float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'average_precision_execute':float(average_precision_score(y, p)) if (y == 1).any() else None,
            'log_loss':float(log_loss(y, p, labels=[0, 1]))}


def counterfactual_metrics(metadata: list[dict], probabilities: np.ndarray, threshold=.5) -> dict:
    groups = {}
    for index, row in enumerate(metadata):
        pid = row['counterfactual_pair_id']
        if pid:
            groups.setdefault((row['trajectory_id'], pid), []).append((index, row))
    results = []
    for (trajectory, pid), pair in sorted(groups.items()):
        pair.sort(key=lambda entry: entry[1]['state_index'])
        if (len(pair) != 2 or [r['label'] for _, r in pair] != ['EXECUTE', 'SKIP']
                or pair[0][1]['state_index'] >= pair[1][1]['state_index']
                or pair[0][1]['embedding_keys']['candidate_subquery'] != pair[1][1]['embedding_keys']['candidate_subquery']):
            raise ValueError('Invalid or non-identical-wording validation counterfactual pair')
        early, late = (float(probabilities[index]) for index, _ in pair)
        results.append({'trajectory_id':trajectory, 'pair_id':pid,
                        'earlier_example_id':pair[0][1]['example_id'], 'later_example_id':pair[1][1]['example_id'],
                        'earlier_execute_probability':early, 'later_execute_probability':late,
                        'earlier_correct':early >= threshold, 'later_correct':late < threshold,
                        'both_correct':early >= threshold and late < threshold,
                        'probability_decreased':early - late > 1e-8,
                        'probability_change_later_minus_earlier':late - early})
    n = len(results)
    mean = lambda key: float(np.mean([r[key] for r in results])) if n else None
    return {'n_pairs':n, 'probability_decrease_tolerance':1e-8,
            'both_correct_count':sum(r['both_correct'] for r in results),
            'both_correct_fraction':mean('both_correct'), 'earlier_execute_accuracy':mean('earlier_correct'),
            'later_skip_accuracy':mean('later_correct'),
            'mean_earlier_execute_probability':mean('earlier_execute_probability'),
            'mean_later_execute_probability':mean('later_execute_probability'),
            'probability_decreased_count':sum(r['probability_decreased'] for r in results),
            'pairs':results}


def verify_inputs(prepared: Path):
    train = load_prepared(prepared, 'train')
    validation = load_prepared(prepared, 'validation')
    contract = json.loads((prepared/'contract.json').read_text())
    split = json.loads((prepared/'split.json').read_text())
    if (contract['feature_names'] != list(FEATURE_NAMES) or contract['text_fields'] != list(TEXT_FIELDS)
            or contract['labels'] != LABELS or contract['embedding']['dimensions'] != 1536
            or split != contract['split']):
        raise ValueError('Prepared contract differs from locked input definition')
    observed = {}
    for name, (X, y, metadata), n in [('train', train, 320), ('validation', validation, 80)]:
        if X.shape != (n, 4623) or y.shape != (n,) or not np.isfinite(X).all() or set(y) != {0, 1}:
            raise ValueError('Unexpected prepared shapes, values or labels')
        if not np.array_equal(y, [LABELS[r['label']] for r in metadata]):
            raise ValueError('Metadata/target alignment mismatch')
        observed[name] = {r['trajectory_id'] for r in metadata}
        if observed[name] != set(split[name]):
            raise ValueError('Metadata/split mismatch')
        for tid in observed[name]:
            group = [r for r in metadata if r['trajectory_id'] == tid]
            if len(group) != 20 or len({r['question_id'] for r in group}) != 1:
                raise ValueError('Incomplete trajectory')
            if sorted(r['state_index'] for r in group) != [s for s in range(5) for _ in range(4)]:
                raise ValueError('Incomplete states')
        # Validate both sides of every linked pair before optimization.
        counterfactual_metrics(metadata, np.full(n, .5))
    if observed['train'] & observed['validation']:
        raise ValueError('Trajectory overlap')
    if {r['question_id'] for r in train[2]} & {r['question_id'] for r in validation[2]}:
        raise ValueError('Question overlap')
    return train, validation, contract


def train_mlp(X_train, y_train, X_val, y_val, cfg):
    set_seed(cfg.seed)
    model = build_model(cfg)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    tx, ty = torch.from_numpy(X_train.copy()).float(), torch.from_numpy(y_train.copy()).float()
    vx, vy = torch.from_numpy(X_val.copy()).float(), torch.from_numpy(y_val.copy()).float()
    loader = DataLoader(TensorDataset(tx, ty), batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed), num_workers=0)
    best_loss, best_state, best_epoch, stale = float('inf'), None, 0, 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        batch_loss = 0.
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item() * len(y)
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(tx).squeeze(-1), ty).item()
            validation_loss = criterion(model(vx).squeeze(-1), vy).item()
        if not np.isfinite([train_loss, validation_loss]).all():
            raise ValueError('Non-finite loss; abort experiment without changing configuration')
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        history.append({'epoch':epoch, 'train_loss':train_loss, 'validation_loss':validation_loss,
                        'training_minibatch_loss_with_dropout':batch_loss/len(ty), 'best_so_far':improved})
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_p = torch.sigmoid(model(tx).squeeze(-1)).numpy()
        validation_p = torch.sigmoid(model(vx).squeeze(-1)).numpy()
    return model, history, best_epoch, train_p, validation_p


def fit_linear(X_train, y_train, X_val, cfg):
    # One fixed C and solver for all three linear fits; no extra feature scaling.
    model = LogisticRegression(C=cfg.logistic_C, solver='lbfgs', max_iter=cfg.logistic_max_iter,
                               class_weight=None, random_state=cfg.seed)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
    return model, probability, [str(w.message) for w in caught]


def plot_history(history, best_epoch, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Train (eval mode)')
    ax.plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], label='Validation')
    ax.axvline(best_epoch, color='gray', linestyle='--', label=f'Best epoch {best_epoch}')
    ax.set(xlabel='Epoch', ylabel='BCE loss', title='Fixed MLP baseline: loss history')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(cfg: TrainingConfig, config_path: Path) -> Path:
    # All preflight checks happen before creating an experiment or fitting models.
    (tx, ty, tm), (vx, vy, vm), contract = verify_inputs(cfg.prepared)
    if cfg.input_dimensions != tx.shape[1]:
        raise ValueError('Configured input dimension does not match prepared data')
    input_hashes = {p.name:sha(p) for p in sorted(cfg.prepared.iterdir()) if p.is_file()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = cfg.output_root/f'baseline-v1-{stamp}'
    if out.resolve().is_relative_to(cfg.prepared.resolve()):
        raise ValueError('Training output must not be inside prepared data')
    out.mkdir(parents=True, exist_ok=False)
    exact = cfg.model_dump(mode='json')
    exact.update(device='cpu', torch_num_threads=1, deterministic_algorithms=True,
                 loss='BCEWithLogitsLoss', optimizer='AdamW', class_weight=None,
                 early_stopping_min_delta=0, logistic_solver='lbfgs', logistic_class_weight=None,
                 additional_scaling=False, linear_threadpool_limit=1,
                 counterfactual_probability_decrease_tolerance=1e-8)
    save_json(out/'config.json', exact)
    save_json(out/'validation_metadata.json', vm)
    save_json(out/'split.json', contract['split'])
    save_json(out/'manifest.json', {'status':'running','prepared_directory':str(cfg.prepared.resolve()),
                                   'prepared_sha256':input_hashes})
    model, history, best_epoch, train_p, mlp_p = train_mlp(tx, ty, vx, vy, cfg)
    torch.save({'state_dict':model.state_dict(), 'config':exact, 'best_epoch':best_epoch,
                'best_validation_loss':history[best_epoch-1]['validation_loss']}, out/'best_mlp.pt')
    save_json(out/'history.json', history)
    majority = int(np.bincount(ty).argmax())
    probabilities = {'mlp':mlp_p, 'majority':np.full(len(vy), float(majority))}
    linear_info = {}
    for name, columns in [('logistic_full',slice(None)), ('logistic_candidate_only',slice(3072,4608)),
                          ('logistic_numerical_only',slice(4608,4623))]:
        fitted, p, notices = fit_linear(tx[:,columns], ty, vx[:,columns], cfg)
        probabilities[name] = p
        linear_info[name] = {'convergence_warnings':notices, 'iterations':fitted.n_iter_.tolist(),
                             'input_dimensions':tx[:,columns].shape[1]}
        np.savez_compressed(out/f'{name}.npz', coef=fitted.coef_, intercept=fitted.intercept_, classes=fitted.classes_)
    evaluations = {name:classification_metrics(vy,p,cfg.threshold) for name,p in probabilities.items()}
    pairs = {name:counterfactual_metrics(vm,p,cfg.threshold) for name,p in probabilities.items()}
    best = history[best_epoch-1]
    save_json(out/'metrics.json', {'best_epoch':best_epoch, 'epochs_run':len(history),
                                  'restored_train_loss':best['train_loss'],
                                  'restored_validation_loss':best['validation_loss'],
                                  'restored_train_metrics':classification_metrics(ty,train_p,cfg.threshold),
                                  'validation':evaluations})
    save_json(out/'counterfactual.json',pairs)
    save_json(out/'baselines.json', {'majority_class_from_train':majority,'linear_fit_details':linear_info,
                                   'validation':{k:v for k,v in evaluations.items() if k!='mlp'}})
    # Arrays contain no IDs or reasons; index-aligned metadata is a separate file.
    np.savez_compressed(out/'validation_predictions.npz', y_true=vy,
                        **{name+'_probability':p for name,p in probabilities.items()},
                        **{name+'_prediction':(p>=cfg.threshold).astype(np.int64) for name,p in probabilities.items()})
    plot_history(history,best_epoch,out/'loss_curves.png')
    if input_hashes != {p.name:sha(p) for p in sorted(cfg.prepared.iterdir()) if p.is_file()}:
        raise RuntimeError('Prepared files changed during experiment')
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    code_paths = [Path(__file__), Path('src/adr/mlp/prepare.py'), config_path, Path('pyproject.toml')]
    save_json(out/'manifest.json', {'status':'complete', 'created_utc':stamp,
        'prepared_directory':str(cfg.prepared.resolve()), 'prepared_sha256':input_hashes,
        'prepared_unchanged':True, 'source_code_sha256':{str(p):sha(p) for p in code_paths},
        'git_commit':commit, 'working_tree_note':'Code hashes include uncommitted implementation; no commit created.',
        'versions':{'python':platform.python_version(), 'torch':torch.__version__,
                    'numpy':np.__version__, 'scikit_learn':sklearn.__version__},
        'platform':platform.platform(), 'device':'cpu',
        'command':f'PYTHONPATH=src .venv/bin/python -m adr.mlp.train --config {config_path}',
        'selection':'Single fixed configuration. Best epoch selected by validation loss, threshold fixed at 0.5; no architecture/parameter search.',
        'evaluation_limit':'Four synthetic validation trajectories used for early stopping; not an independent test set or real DRB evaluation.',
        'artifacts_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.name!='manifest.json'}})
    print(json.dumps({'output':str(out.resolve()),'best_epoch':best_epoch,'epochs_run':len(history),
                      'train_loss':best['train_loss'],'validation_loss':best['validation_loss'],
                      'validation':evaluations, 'counterfactual':{k:{f:v for f,v in r.items() if f!='pairs'} for k,r in pairs.items()}},indent=2))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/mlp_baseline.yaml'))
    args=parser.parse_args()
    cfg=TrainingConfig.model_validate(yaml.safe_load(args.config.read_text()))
    run(cfg,args.config)


if __name__=='__main__':
    main()
