"""Approved development-only factorial; --preflight needs only Python stdlib.

No hyperparameter CLI, held-out evaluation, or automatic model search.
Frozen evaluator and reconstruction implementations are imported unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = "fa746853ba007552bb1d5a43340d91f6796096b1"
REF = "44714c2e0be958fe796a8fd4bdbc220dae3c23dd"
R3 = ROOT / "results/real_cellseg_round3_thresholding_20260916"
R5 = ROOT / "results/real_cellseg_round5_corrected_baseline_20260917"
R6 = ROOT / "results/real_cellseg_round6_paper_downstream_20260917"
PY_VERSIONS = {"torch": "2.10.0", "gpytorch": "1.15.1", "qpytorch": "0.2",
               "numpy": "1.26.4", "scikit-image": "0.25.2", "pandas": "2.3.3"}
R_VERSIONS = {"R": "4.5.1", "EBImage": "4.52.0", "RobustGaSP": "0.6.8",
              "magick": "2.9.1", "pracma": "2.4.6"}
FIT = dict(seed=0, train_iters=75, lr=0.1, max_points=3000, nu=2.5,
           predict_chunk=8192, standardize=True, exact_logdet=True, device="cpu")


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def table(path, values):
    if not values:
        return
    keys = list(dict.fromkeys(k for v in values for k in v))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(values)


def command(args, **kw):
    return subprocess.check_output([str(x) for x in args], text=True, **kw).strip()


def digest(path, algorithm="sha256"):
    return hashlib.new(algorithm, Path(path).read_bytes()).hexdigest()


def preflight(args):
    errors = []
    record = dict(base=BASE, reference=REF, python=platform.python_version(),
                  platform=platform.platform(), packages={}, R_packages={}, inputs=[],
                  protocol=FIT, new_results=0)
    for repo, expected in [(ROOT, BASE), (args.reference, REF)]:
        try:
            head = command(["git", "-C", repo, "rev-parse", "HEAD"])
            # Added experiment commits are allowed; all frozen tracked bytes must match.
            command(["git", "-C", repo, "merge-base", "--is-ancestor", expected, head])
            changes = command(["git", "-C", repo, "diff", "--name-status", expected])
            changed_old = [x for x in changes.splitlines() if not x.startswith("A\t")]
            if changed_old:
                errors.append(f"Frozen tracked files changed in {repo}: {changed_old}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            errors.append(f"Checkpoint verification failed: {repo}: {e}")
    manifest = [x for x in rows(R3 / "image_manifest.csv") if x["role"] == "development"]
    if {(x["dataset"], x["image"]) for x in manifest} != {
            ("nuclei", "nuclei_figure_1"), ("whole_cell", "whole_cell_figure_1")}:
        errors.append("Unexpected development image selection")
    for m in manifest:
        for column, hash_column in [("path_image", "md5_image"), ("path_gt", "md5_gt")]:
            p = ROOT / m[column]
            ok = p.exists() and digest(p, "md5") == m[hash_column]
            record["inputs"].append(dict(path=m[column], matches_frozen_md5=ok,
                                         sha256=digest(p) if p.exists() else None))
            if not ok:
                errors.append(f"Input checksum mismatch: {p}")
    if record["python"] != "3.10.19":
        errors.append(f"Python {record['python']} != frozen 3.10.19")
    for pkg in list(PY_VERSIONS) + ["scipy", "imageio", "tifffile", "Pillow", "matplotlib"]:
        try:
            actual = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            actual = None
        record["packages"][pkg] = actual
        if actual is None or (pkg in PY_VERSIONS and actual.split("+")[0] != PY_VERSIONS[pkg]):
            errors.append(f"Python package {pkg}: found {actual}, required {PY_VERSIONS.get(pkg, 'installed')}")
    if not shutil.which(args.rscript):
        errors.append(f"Rscript unavailable: {args.rscript}")
    else:
        expr = 'cat("R=",as.character(getRversion()),"\\n",sep="");' + \
            'for(p in c("EBImage","RobustGaSP","magick","pracma")) ' + \
            'cat(p,"=",if(requireNamespace(p,quietly=TRUE)) as.character(packageVersion(p)) else "MISSING","\\n",sep="")'
        try:
            found = dict(line.split("=", 1) for line in command([args.rscript, "-e", expr]).splitlines())
            record["R_packages"] = found
            for pkg, version in R_VERSIONS.items():
                if found.get(pkg) != version:
                    errors.append(f"R package {pkg}: {found.get(pkg)} != {version}")
        except subprocess.SubprocessError as e:
            errors.append(f"R dependency probe failed: {e}")
    if args.qep_cache_dir:
        cache_manifest = json.loads((HERE / "historical_cache_manifest.json").read_text())
        record["qep_cache_files"] = []
        for name, expected in cache_manifest["entries"].items():
            path = args.qep_cache_dir / name
            matched = path.is_file() and digest(path) == expected["file_sha256"]
            record["qep_cache_files"].append(dict(path=str(path), checksum_matched=matched))
            if not matched:
                errors.append(f"Missing or changed historical QEP cache: {path}")
    record["runner_sha256"] = {p.name: digest(p) for p in HERE.glob("*") if p.suffix in (".py", ".R", ".json")}
    record["errors"] = errors
    record["status"] = "BLOCKED" if errors else "READY_FOR_ANCHOR_CHECKS"
    dump(args.out / "preflight.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return manifest, record


def run(args, manifest):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from py_core.foreground_threshold import apply_rule, foreground_metrics
    from py_core.instance_separation import separate_instances
    from py_core.segmentation_eval import (load_gray_image, load_instance_mask,
        evaluate_instances, classify_failure_modes, compute_ious_fast)
    from py_core.segmentation_pipeline import compute_tiling, smooth_tile

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    all_rows, states, comparisons, anchors = [], {}, [], []
    if (args.out / "metrics.csv").exists():
        raise RuntimeError("Output contains results; choose a fresh output directory")

    def rcall(script, argv, log):
        with open(log, "w") as f:
            subprocess.run([args.rscript, str(HERE / script), *map(str, argv)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    def array_hash(a):
        return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()

    def score(ds, method, downstream, reconstruction, seconds, gt, work):
        sha = array_hash(reconstruction)
        t0 = time.perf_counter()
        if downstream == "P":
            rcall("paper_downstream.R", [ds, ROOT / work["manifest"]["path_image"],
                  work["csv"], work["dir"], method, args.reference],
                  work["dir"] / f"{method}_P.log")
            binary = np.loadtxt(work["dir"] / f"{method}_combined_binary.csv", delimiter=",") > 0
            labels = np.loadtxt(work["dir"] / f"{method}_labels_final.csv", delimiter=",")
            summary = rows(work["dir"] / f"{method}_summary.csv")[0]
            extra = dict(n_markers=None, marker_definition="EBImage implicit; count not substituted",
                         n_precleanup=int(summary["n_labels_precleanup"]),
                         n_outlier_tiles=int(summary["n_outlier_tiles"]))
        else:
            tr = apply_rule(reconstruction, "li", {})
            if tr.failed:
                raise RuntimeError("Frozen Li threshold failed")
            binary = tr.mask
            sep = separate_instances(binary, marker_mode="peak",
                min_distance=15 if ds == "nuclei" else 9,
                remove_size_threshold=50, rescue_empty_components=True)
            labels = sep.instance_mask
            pre = watershed(-sep.dist_map, markers=sep.marker_image, mask=binary, connectivity=1)
            extra = dict(threshold=tr.threshold, n_markers=sep.n_markers,
                         marker_definition="explicit peak plus rescue", n_rescued=sep.n_components_rescued,
                         n_precleanup=int(np.unique(pre[pre > 0]).size))
            comp, nc = ndi.label(binary)
            marker_counts = np.bincount(comp[sep.marker_image > 0], minlength=nc + 1)
            table(work["dir"] / f"{method}_C_markers_per_component.csv",
                  [dict(component=i, n_markers=int(marker_counts[i])) for i in range(1, nc+1)])
            np.savez_compressed(work["dir"] / f"{method}_C_markers.npz", markers=sep.marker_image)
        elapsed = time.perf_counter() - t0
        if labels.shape != gt.shape or not np.isfinite(labels).all() or np.any(labels != np.floor(labels)):
            raise RuntimeError("Invalid label geometry/values")
        labels = labels.astype(np.int32)
        if labels.min() < 0 or array_hash(reconstruction) != sha:
            raise RuntimeError("Negative labels or mutated reconstruction")
        np.savez_compressed(work["dir"] / f"{method}_{downstream}_segmentation.npz",
                            labels=labels, foreground=binary)
        ev = evaluate_instances(gt, labels)
        failures = classify_failure_modes(gt, labels)
        row = dict(dataset=ds, method=method, downstream=downstream, input_sha256=sha,
                   ap50=ev.ap(.5), ap75=ev.ap(.75), n_gt=ev.n_true, n_pred=ev.n_pred,
                   mean_best_iou=ev.mean_matched_iou, **foreground_metrics(binary, gt > 0),
                   **{k: failures[k] for k in ("merged", "split", "missed", "spurious")},
                   fg_components_4=int(ndi.label(binary)[1]), runtime_reconstruction_s=seconds,
                   runtime_downstream_s=elapsed,
                   runtime_total_s=None if seconds is None else seconds+elapsed,
                   reconstruction_source="verified historical cache" if seconds is None else "generated in this run",
                   **extra)
        for threshold, suffix in [(.5, "50"), (.75, "75")]:
            for metric in ("tp", "fp", "fn"):
                row[metric+suffix] = ev.per_threshold[threshold][metric]
        ious = compute_ious_fast(gt, labels).to_numpy()
        matched = ious.max(axis=1) >= .5 if ious.shape[1] else np.zeros(ev.n_true, bool)
        split = np.zeros(ev.n_true, bool)
        # Per-GT identity diagnostics use the exact frozen 25% split definition.
        for i, label in enumerate(np.unique(gt[gt > 0])):
            overlap = np.bincount(labels[gt == label])
            split[i] = (overlap[1:] / overlap.sum() >= .25).sum() >= 2
        assert int(split.sum()) == failures["split"]
        states[ds, method, downstream] = (matched, split)
        all_rows.append(row)
        table(args.out / "metrics.csv", all_rows)
        print(f"{ds} {method}-{downstream}: AP50={row['ap50']:.8f}", flush=True)
        return row

    def check_anchor(row, old):
        keys = [f"{k}{s}" for s in ("50", "75") for k in ("tp", "fp", "fn")]
        bad = {k: [row[k], int(old[k])] for k in keys if row[k] != int(old[k])}
        # Foreground and failure counts also protect against accidental metric parity.
        for k in ("merged", "split", "missed", "spurious"):
            if row[k] != int(old[k]):
                bad[k] = [row[k], int(old[k])]
        if abs(row["fg_dice"] - float(old["fg_dice"])) > 1e-10:
            bad["fg_dice"] = [row["fg_dice"], float(old["fg_dice"])]
        anchors.append(dict(dataset=row["dataset"], method=row["method"],
                            downstream=row["downstream"], passed=not bad, differences=bad))
        dump(args.out / "anchor_checks.json", anchors)
        if bad:
            raise RuntimeError(f"Anchor mismatch; stop without retuning: {anchors[-1]}")

    prepared = {}
    # Complete both images' non-QEP anchors before any QEP training.
    for m in manifest:
        ds = m["dataset"]
        folder = args.out / ds
        folder.mkdir(exist_ok=True)
        raw = load_gray_image(str(ROOT / m["path_image"]))
        gt = load_instance_mask(str(ROOT / m["path_gt"]))
        assert raw.shape == gt.shape == (int(m["height"]), int(m["width"]))
        assert np.unique(gt[gt > 0]).size == int(m["n_gt_instances"])
        np.savetxt(folder / "raw.csv", raw, delimiter=",", fmt="%.17g")
        t0 = time.perf_counter()
        rcall("restore_fastgp.R", [args.reference, folder / "raw.csv",
              ROOT / f"audits/parity/real_{ds}_r_params.csv", folder / "fastgp.csv"],
              folder / "restore_fastgp.log")
        fast_seconds = time.perf_counter() - t0
        recon = np.loadtxt(folder / "fastgp.csv", delimiter=",")
        assert recon.shape == raw.shape and np.isfinite(recon).all()
        for method, arr, sec in [("raw", raw, 0.0), ("fastgp", recon, fast_seconds)]:
            np.savez_compressed(folder / f"{method}_reconstruction.npz", predmean=arr)
            work = dict(manifest=m, csv=folder/f"{method}.csv", dir=folder)
            for down in ("P", "C"):
                row = score(ds, method, down, arr, sec, gt, work)
                cell = "B" if method == "raw" and down == "P" else "A" if method == "raw" else "D" if down == "P" else None
                if cell:
                    old = next(x for x in rows(R6 / "two_by_two_metrics.csv") if x["dataset"] == ds and x["cell"] == cell)
                    check_anchor(row, old)
        prepared[ds] = (m, folder, raw, gt)

    for ds, (m, folder, raw, gt) in prepared.items():
        geom = compute_tiling(*raw.shape)
        for method, q in [("q2", 2.0), ("q15", 1.5)]:
            if args.qep_cache_dir:
                from cache_io import load_verified_cache
                print(f"{ds} {method}: loading verified historical reconstruction; no training", flush=True)
                recon, provenance = load_verified_cache(args.qep_cache_dir, ds, method, raw.shape)
                dump(folder / f"{method}_cache_provenance.json", provenance)
                np.savez_compressed(folder / f"{method}_reconstruction.npz", predmean=recon)
                np.savetxt(folder / f"{method}.csv", recon, delimiter=",", fmt="%.17g")
                work = dict(manifest=m, csv=folder/f"{method}.csv", dir=folder)
                c = score(ds, method, "C", recon, None, gt, work)
                old_name = "qep_q2" if q == 2 else "qep_q1.5"
                old = next(x for x in rows(R5 / "development_segmentation_metrics.csv")
                           if x["dataset"] == ds and x["method"] == old_name)
                check_anchor(c, old)
                score(ds, method, "P", recon, None, gt, work)
                continue
            recon = np.zeros_like(raw)
            diagnostics = []
            t0 = time.perf_counter()
            index = 0
            for i in range(geom.num_pieces_x):
                for j in range(geom.num_pieces_y):
                    x, y = i*geom.crop_width, j*geom.crop_height
                    w = raw.shape[1]-x if i == geom.num_pieces_x-1 else geom.crop_width
                    h = raw.shape[0]-y if j == geom.num_pieces_y-1 else geom.crop_height
                    tile, diag = smooth_tile(raw[y:y+h, x:x+w].copy(), family="qep", q=q,
                        nu=2.5, train_iters=75, lr=.1, max_points=3000,
                        rng=np.random.default_rng([0, index]), dtype=torch.float64,
                        standardize=True, predict_chunk=8192, torch_seed=index,
                        exact_logdet=True, device="cpu")
                    diag.update(tile_index=index, x=x, y=y, width=w, height=h)
                    diagnostics.append(diag)
                    dump(folder / f"{method}_fit.json", diagnostics)
                    if diag["n_effective_steps"] != 75 or diag["n_nonfinite_loss"] or diag["n_exceptions"]:
                        raise RuntimeError("Training incomplete; no silent optimizer repair")
                    recon[y:y+h, x:x+w] = tile
                    np.savez_compressed(folder / f"{method}_tile{index:02d}.npz", predmean=tile)
                    index += 1
                    print(f"{ds} {method}: trained tile {index}/{geom.num_pieces_x*geom.num_pieces_y}", flush=True)
            seconds = time.perf_counter() - t0
            # Historical cache contract: float32 storage, float64 downstream loading.
            recon = recon.astype(np.float32).astype(np.float64)
            assert np.isfinite(recon).all()
            np.savez_compressed(folder / f"{method}_reconstruction.npz", predmean=recon)
            np.savetxt(folder / f"{method}.csv", recon, delimiter=",", fmt="%.17g")
            work = dict(manifest=m, csv=folder/f"{method}.csv", dir=folder)
            c = score(ds, method, "C", recon, seconds, gt, work)
            old_name = "qep_q2" if q == 2 else "qep_q1.5"
            old = next(x for x in rows(R5 / "development_segmentation_metrics.csv")
                       if x["dataset"] == ds and x["method"] == old_name)
            check_anchor(c, old)
            score(ds, method, "P", recon, seconds, gt, work)

    for ds in prepared:
        lookup = {(r["method"], r["downstream"]): r for r in all_rows if r["dataset"] == ds}
        for down in ("P", "C"):
            candidate = lookup["q15", down]
            for control, cd in dict.fromkeys([("fastgp", "P"), ("raw", down), ("fastgp", down), ("q2", down)]):
                ref = lookup[control, cd]
                cm, cs = states[ds, "q15", down]
                rm, rs = states[ds, control, cd]
                comparisons.append(dict(dataset=ds, candidate=f"q15-{down}", control=f"{control}-{cd}",
                    delta_ap50=candidate["ap50"]-ref["ap50"], delta_ap75=candidate["ap75"]-ref["ap75"],
                    newly_split=int((cs & ~rs).sum()), newly_split_fraction=float((cs & ~rs).mean()),
                    lost_previously_matched=int((rm & ~cm).sum()),
                    lost_gt_ids=(np.flatnonzero(rm & ~cm)+1).tolist(),
                    newly_split_gt_ids=(np.flatnonzero(cs & ~rs)+1).tolist()))
    assert len(all_rows) == 16
    for ds in prepared:
        for method in ("raw", "fastgp", "q2", "q15"):
            hashes = {r["input_sha256"] for r in all_rows if r["dataset"] == ds and r["method"] == method}
            assert len(hashes) == 1
    dump(args.out / "comparisons.json", comparisons)
    table(args.out / "comparisons.csv", comparisons)
    conditional = []
    for ds in prepared:
        lookup = {(r["method"], r["downstream"]): r for r in all_rows if r["dataset"] == ds}
        for method in ("raw", "fastgp", "q2", "q15"):
            conditional.append(dict(dataset=ds, method=method,
                delta_ap50_C_minus_P=lookup[method, "C"]["ap50"]-lookup[method, "P"]["ap50"],
                delta_ap75_C_minus_P=lookup[method, "C"]["ap75"]-lookup[method, "P"]["ap75"]))
    table(args.out / "downstream_contrasts.csv", conditional)
    gates = []
    for ds in prepared:
        lookup = {(r["method"], r["downstream"]): r for r in all_rows if r["dataset"] == ds}
        for down in ("P", "C"):
            cand = lookup["q15", down]
            best = max((lookup[x, down] for x in ("raw", "fastgp", "q2")), key=lambda x: x["ap50"])
            relevant = [c for c in comparisons if c["dataset"] == ds and c["candidate"] == f"q15-{down}"]
            # Report safety vs all controls; conservatively require it for all.
            # This explicit operational convention is fixed before any scores exist.
            gain_original = cand["ap50"]-lookup["fastgp", "P"]["ap50"]
            gain_best = cand["ap50"]-best["ap50"]
            safe = all(c["delta_ap75"] >= -.01 and c["newly_split_fraction"] <= .01 for c in relevant)
            gates.append(dict(dataset=ds, candidate=f"q15-{down}",
                best_same_downstream_gaussian_or_raw=best["method"],
                ap50_gain_vs_original=gain_original, ap50_gain_vs_best_control=gain_best,
                safety_passed_all_controls=safe,
                engineering_gate_passed=gain_original >= .02 and gain_best >= .01 and safe))
    dump(args.out / "engineering_gates.json", gates)
    report = ["# QEP reconstruction × downstream development pilot", "",
              "All 16 frozen combinations completed. These are development-image observations, not generalization or significance results.",
              "", "| Dataset | Reconstruction | Downstream | AP50 | AP75 | Merge | Split | FG Dice |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in all_rows:
        report.append(f"| {r['dataset']} | {r['method']} | {r['downstream']} | {r['ap50']:.6f} | {r['ap75']:.6f} | {r['merged']} | {r['split']} | {r['fg_dice']:.6f} |")
    report += ["", "AP is the frozen TP/(TP+FP+FN) metric. Q2 is the same QEP implementation at the Gaussian limit.",
        "", "Q15/Q2 differences include q-dependent hyperparameter fitting and likelihood effects; they are not a direct fixed-parameter mean effect.",
        "", "See comparisons.csv for all control contrasts, AP75 changes, new splits and lost matches; downstream_contrasts.csv for C minus P.",
        "", "Engineering gates are descriptive predeclared continuation criteria. AP75/split safety is conservatively required against all listed controls. No p-values are computed.",
        "", "FastGP runtime excludes the historical optimizer: its frozen R-side fit is restored. Runtime columns cannot support a full-fit speed ranking.",
        "", "When historical QEP caches are used, reconstruction/total runtime is unavailable (null), not zero. Historical full-pipeline runtime is retained separately in cache provenance.",
        "", "No automatic follow-up search or held-out evaluation is authorized by these results."]
    (args.out / "REPORT.md").write_text("\n".join(report) + "\n")
    dump(args.out / "completion.json", dict(status="COMPLETE", n_results=16,
        inference="Development images only; q differences include fitting and likelihood effects",
        runtime_note="FastGP reconstruction restores frozen parameters; optimizer time excluded; no fair full-fit runtime ranking"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--qep-cache-dir", type=Path,
                   help="Original Round-3 masks directory; all four caches must match the inspected manifest")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    args.reference = args.reference.resolve()
    if args.qep_cache_dir:
        args.qep_cache_dir = args.qep_cache_dir.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, record = preflight(args)
    if record["errors"]:
        return 2
    if args.preflight:
        return 0
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        run(args, manifest)
    except Exception as e:
        dump(args.out / "STOPPED.json", dict(status="STOPPED", error=str(e), interpretation_allowed=False))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
