"""Execute frozen metric definitions without unrelated visualization/ML imports."""
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
AP_THRESHOLDS = np.round(np.arange(.5, .9+1e-9, .05), 2)
for relative, names in [
    ('py_core/metrics.py', {'compute_ious', 'compute_ap_from_ious'}),
    ('py_core/segmentation_eval.py', {'compute_ious_fast', 'check_iou_parity',
     'EvalResult', 'evaluate_instances', 'classify_failure_modes'}),
    ('py_core/foreground_threshold.py', {'foreground_metrics'}),
]:
    path = ROOT/relative
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(nodes) != len(names):
        raise RuntimeError(f'Frozen definitions missing: {relative}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals())


def gray(path):
    with Image.open(path) as image:
        a = np.asarray(image)
        return (a[..., 0] if a.ndim == 3 else a).astype(np.float64)


def ground_truth(path):
    a = gray(path)
    return np.searchsorted(np.unique(a), a).astype(np.int32)


def validate_labels(a, shape):
    if a.shape != shape or not np.isfinite(a).all() or np.any(a < 0) or np.any(a != np.floor(a)):
        raise RuntimeError('Invalid label shape/values')
    return a.astype(np.int32)


def canonical(labels):
    """Same foreground and same partition, permitting only label renumbering."""
    positive = labels.ravel() > 0
    _, first, inverse = np.unique(labels.ravel()[positive], return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))+1
    result = np.zeros(labels.size, dtype=np.int32)
    result[positive] = order[inverse]
    return result.reshape(labels.shape)


def score(gt, labels, foreground):
    labels = validate_labels(labels, gt.shape)
    if foreground.shape != gt.shape:
        raise RuntimeError('Invalid foreground shape')
    ev = evaluate_instances(gt, labels)
    failures = classify_failure_modes(gt, labels)
    row = dict(ap50=ev.ap(.5), ap75=ev.ap(.75), n_gt=ev.n_true, n_pred=ev.n_pred,
               mean_best_iou=ev.mean_matched_iou, **foreground_metrics(foreground, gt > 0),
               **{k: failures[k] for k in ('merged', 'split', 'missed', 'spurious')},
               fg_components_4=int(ndimage.label(foreground)[1]))
    for th, suffix in ((.5, '50'), (.75, '75')):
        for key in ('tp', 'fp', 'fn'):
            row[key+suffix] = ev.per_threshold[th][key]
    ious = compute_ious_fast(gt, labels).to_numpy()
    matched = ious.max(axis=1) >= .5 if ious.shape[1] else np.zeros(ev.n_true, bool)
    split = []
    for label in np.unique(gt[gt > 0]):
        overlaps = np.bincount(labels[gt == label])
        split.append(bool((overlaps[1:]/overlaps.sum() >= .25).sum() >= 2))
    if sum(split) != failures['split']:
        raise RuntimeError('Per-GT split parity failed')
    return row, {'matched': matched, 'split': np.asarray(split)}


def anchor_differences(row, old):
    integers = ['n_gt', 'n_pred', 'merged', 'split', 'missed', 'spurious', 'fg_components_4']
    integers += [k+s for k in ('tp', 'fp', 'fn') for s in ('50', '75')]
    floats = ['ap50', 'ap75', 'mean_best_iou', 'fg_dice', 'fg_iou', 'fg_precision', 'fg_recall']
    bad = {k: [row[k], int(old[k])] for k in integers if row[k] != int(old[k])}
    bad.update({k: [row[k], float(old[k])] for k in floats if abs(row[k]-float(old[k])) > 1e-10})
    return bad


def identity_changes(before, after):
    a, b = before['matched'], after['matched']
    flags = {'gained_matched': ~a & b, 'lost_matched': a & ~b,
             'new_split_previously_matched': a & ~before['split'] & after['split']}
    return {key: {'count': int(value.sum()), 'gt_labels': (np.flatnonzero(value)+1).tolist()}
            for key, value in flags.items()}
