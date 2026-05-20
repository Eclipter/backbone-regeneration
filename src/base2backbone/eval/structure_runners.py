"""Structure export runners for baselines and the trained model."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm

from ..data import parse_dna
from ..runtime import PROGRESS_BAR_COLOR
from ..inference import (
    _build_output_universe as build_output_universe,
    _predict_backbone_from_chain_records as predict_backbone_from_chain_records,
    write_structure,
)
from .structure_rmsd import median_backbone_rmsd_vs_predictions

if TYPE_CHECKING:
    from .knn_baseline import KnnBaselineState


class BackboneSampler(Protocol):
    def sample(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class CheckpointSamplerAdapter:
    """Wrap checkpoint ``sample``; ``num_timesteps=None`` uses ``model.hparams`` default."""

    def __init__(self, checkpoint_model, num_timesteps: int | None):
        self.checkpoint_model = checkpoint_model
        self.num_timesteps = num_timesteps

    def sample(self, batch):
        if self.num_timesteps is None:
            return self.checkpoint_model.sample(batch)
        return self.checkpoint_model.sample(
            batch,
            num_timesteps=self.num_timesteps,
        )


class FixedTorsionSampler:
    """Return constant torsion angles and tau_m for every graph in the batch."""

    def __init__(self, mean_theta: np.ndarray, mean_tau: float):
        self._mean_theta = torch.as_tensor(mean_theta, dtype=torch.float32)
        self._mean_tau = float(mean_tau)

    def sample(self, batch):
        device = batch.torsions.device
        batch_size = int(batch.num_graphs)
        theta = self._mean_theta.to(device).unsqueeze(0).expand(batch_size, -1)
        tau = torch.full(
            (batch_size,),
            self._mean_tau,
            device=device,
            dtype=torch.float32,
        )
        return theta, tau


def export_backbone_pdb(
    input_path: str | Path,
    output_path: str | Path,
    sampler: BackboneSampler,
    device: str,
    *,
    window_size: int,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        _, chain_records = parse_dna(
            str(input_path),
            use_full_nucleotide=False,
            window_size=window_size,
        )
        predictions = predict_backbone_from_chain_records(
            [chain_records],
            sampler,
            device,
        )[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                str(input_path),
                use_full_nucleotide=True,
                window_size=window_size,
            )
        write_structure(
            build_output_universe(full_chain_records, predictions),
            output_path,
        )
    except Exception as exc:
        return {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(exc),
        }

    return {
        'success': True,
        'wall_time_s': time.perf_counter() - t0,
        'returncode': 0,
        'output_pdb': output_path,
        'stdout': '',
        'stderr': '',
    }


def _benchmark_success_result(
    output_pdb: Path,
    t0: float,
    *,
    wall_time_s: float | None = None,
) -> dict[str, Any]:
    return {
        'success': True,
        'wall_time_s': wall_time_s if wall_time_s is not None else time.perf_counter() - t0,
        'returncode': 0,
        'output_pdb': output_pdb,
        'stdout': '',
        'stderr': '',
    }


def run_base2backbone_inference(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    sampler: BackboneSampler,
    device: str,
    window_size: int,
    resume: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / f'{input_path.stem}_base2backbone.pdb'
    t0 = time.perf_counter()

    if resume and output_pdb.is_file() and output_pdb.stat().st_size > 0:
        return _benchmark_success_result(output_pdb, t0, wall_time_s=0.0)

    input_path_str = str(input_path)
    try:
        with torch.inference_mode():
            _, chain_records = parse_dna(
                input_path_str,
                use_full_nucleotide=False,
                window_size=window_size,
            )
            predictions = predict_backbone_from_chain_records(
                [chain_records],
                sampler,
                device,
            )[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                input_path_str,
                use_full_nucleotide=True,
                window_size=window_size,
            )
        write_structure(
            build_output_universe(full_chain_records, predictions),
            output_pdb,
        )
    except Exception as exc:
        return {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(exc),
        }

    return _benchmark_success_result(output_pdb, t0)


def run_base2backbone_inference_batch(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    sampler: BackboneSampler,
    device: str,
    window_size: int,
    window_batch_size: int | None = None,
    resume: bool = True,
) -> dict[str, dict[str, Any]]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[Path, Path, float]] = []
    chain_records_by_structure = []

    for raw_input_path in input_paths:
        input_path = Path(raw_input_path).resolve()
        output_pdb = output_dir / f'{input_path.stem}_base2backbone.pdb'
        t0 = time.perf_counter()
        input_key = str(input_path)

        if resume and output_pdb.is_file() and output_pdb.stat().st_size > 0:
            results[input_key] = _benchmark_success_result(
                output_pdb,
                t0,
                wall_time_s=0.0,
            )
            continue

        try:
            _, chain_records = parse_dna(
                str(input_path),
                use_full_nucleotide=False,
                window_size=window_size,
            )
        except Exception as exc:
            results[input_key] = {
                'success': False,
                'wall_time_s': time.perf_counter() - t0,
                'returncode': -999,
                'output_pdb': None,
                'stdout': '',
                'stderr': repr(exc),
            }
            continue

        pending.append((input_path, output_pdb, t0))
        chain_records_by_structure.append(chain_records)

    if not pending:
        return results

    try:
        with torch.inference_mode():
            predictions_by_structure = predict_backbone_from_chain_records(
                chain_records_by_structure,
                sampler,
                device,
                window_batch_size=window_batch_size,
            )
    except Exception as exc:
        for input_path, _output_pdb, t0 in pending:
            results[str(input_path)] = {
                'success': False,
                'wall_time_s': time.perf_counter() - t0,
                'returncode': -999,
                'output_pdb': None,
                'stdout': '',
                'stderr': repr(exc),
            }
        return results

    for (input_path, output_pdb, t0), predictions in zip(
        pending,
        predictions_by_structure,
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', PDBConstructionWarning)
                _, full_chain_records = parse_dna(
                    str(input_path),
                    use_full_nucleotide=True,
                    window_size=window_size,
                )
            write_structure(
                build_output_universe(full_chain_records, predictions),
                output_pdb,
            )
        except Exception as exc:
            results[str(input_path)] = {
                'success': False,
                'wall_time_s': time.perf_counter() - t0,
                'returncode': -999,
                'output_pdb': None,
                'stdout': '',
                'stderr': repr(exc),
            }
            continue

        results[str(input_path)] = _benchmark_success_result(output_pdb, t0)

    return results


_pool_base2backbone: dict[str, Any] | None = None


def init_base2backbone_inference_worker(
    checkpoint_model,
    device: str,
    window_size: int,
    num_timesteps: int | None = None,
) -> None:
    """Set per-process model/sampler once (ProcessPoolExecutor initializer)."""
    global _pool_base2backbone
    checkpoint_model.eval()
    _pool_base2backbone = {
        'sampler': CheckpointSamplerAdapter(checkpoint_model, num_timesteps),
        'device': device,
        'window_size': int(window_size),
    }


def run_base2backbone_inference_pooled(
    input_path: str | Path,
    output_pdb_dir: str | Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    if _pool_base2backbone is None:
        raise RuntimeError('base2backbone worker pool not initialized')
    ctx = _pool_base2backbone
    return run_base2backbone_inference(
        input_path,
        output_pdb_dir,
        sampler=ctx['sampler'],
        device=ctx['device'],
        window_size=ctx['window_size'],
        resume=resume,
    )


def run_base2backbone_inference_batch_pooled(
    input_paths: list[str | Path],
    output_pdb_dir: str | Path,
    *,
    window_batch_size: int | None = None,
    resume: bool = True,
) -> dict[str, dict[str, Any]]:
    if _pool_base2backbone is None:
        raise RuntimeError('base2backbone worker pool not initialized')
    ctx = _pool_base2backbone
    return run_base2backbone_inference_batch(
        input_paths,
        output_pdb_dir,
        sampler=ctx['sampler'],
        device=ctx['device'],
        window_size=ctx['window_size'],
        window_batch_size=window_batch_size,
        resume=resume,
    )


def run_mean_baseline_structure(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    mean_theta: np.ndarray,
    mean_tau: float,
    device: str,
    window_size: int,
    resume: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / f'{input_path.stem}_mean_baseline.pdb'
    if resume and output_pdb.is_file() and output_pdb.stat().st_size > 0:
        return _benchmark_success_result(output_pdb, time.perf_counter(), wall_time_s=0.0)
    return export_backbone_pdb(
        input_path,
        output_pdb,
        FixedTorsionSampler(mean_theta, mean_tau),
        device,
        window_size=window_size,
    )


def _best_of_k_output_pdb(output_root: Path, k: int, input_path: Path) -> Path:
    return output_root / f'k{k}' / 'pdb' / f'{input_path.stem}_best_of_{k}.pdb'


def _best_of_k_outputs_complete(
    input_path: Path,
    output_root: Path,
    best_of_k_list: list[int],
) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            _best_of_k_output_pdb(output_root, k, input_path)
            for k in best_of_k_list
        )
    )


def _write_best_of_k_snapshot(
    *,
    input_path: Path,
    output_root: Path,
    k: int,
    full_chain_records,
    best_predictions,
) -> Path:
    output_pdb = _best_of_k_output_pdb(output_root, k, input_path)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    write_structure(
        build_output_universe(full_chain_records, best_predictions),
        output_pdb,
    )
    return output_pdb


def run_base2backbone_best_of_k_inference(
    input_path: str | Path,
    output_root: str | Path,
    *,
    sampler: BackboneSampler,
    device: str,
    window_size: int,
    best_of_k_list: list[int],
    window_batch_size: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> dict[int, dict[str, Any]]:
    """Run up to max(k) stochastic samples; export best-vs-ref snapshot at each k."""
    input_path = Path(input_path).resolve()
    output_root = Path(output_root).resolve()
    k_values = sorted({int(k) for k in best_of_k_list if int(k) > 0})
    if not k_values:
        raise ValueError('best_of_k_list must contain at least one positive k')
    k_max = max(k_values)
    k_snapshot_set = set(k_values)
    t0 = time.perf_counter()

    if resume and _best_of_k_outputs_complete(input_path, output_root, k_values):
        results = {}
        for k in k_values:
            output_pdb = _best_of_k_output_pdb(output_root, k, input_path)
            results[k] = _benchmark_success_result(output_pdb, t0, wall_time_s=0.0)
        return results

    try:
        _, chain_records = parse_dna(
            str(input_path),
            use_full_nucleotide=False,
            window_size=window_size,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                str(input_path),
                use_full_nucleotide=True,
                window_size=window_size,
            )
    except Exception as exc:
        failure = {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(exc),
        }
        return {k: failure for k in k_values}

    best_score: float | None = None
    best_predictions = None
    results: dict[int, dict[str, Any]] = {}

    with torch.inference_mode():
        sample_iter = range(1, k_max + 1)
        if show_progress:
            sample_iter = tqdm(
                sample_iter,
                total=k_max,
                desc=f'best-of-k samples ({input_path.stem})',
                leave=False,
                colour=PROGRESS_BAR_COLOR,
            )
        for sample_idx in sample_iter:
            if show_progress and hasattr(sample_iter, 'set_postfix_str'):
                sample_iter.set_postfix_str(f'sample {sample_idx}/{k_max}')
            try:
                predictions = predict_backbone_from_chain_records(
                    [chain_records],
                    sampler,
                    device,
                    window_batch_size=window_batch_size,
                )[0]
            except Exception as exc:
                failure = {
                    'success': False,
                    'wall_time_s': time.perf_counter() - t0,
                    'returncode': -999,
                    'output_pdb': None,
                    'stdout': '',
                    'stderr': repr(exc),
                }
                for k in k_values:
                    if k not in results:
                        results[k] = failure
                return results

            score = median_backbone_rmsd_vs_predictions(
                input_path,
                predictions,
                window_size=window_size,
            )

            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_predictions = predictions

            if sample_idx not in k_snapshot_set or best_predictions is None:
                continue

            try:
                output_pdb = _write_best_of_k_snapshot(
                    input_path=input_path,
                    output_root=output_root,
                    k=sample_idx,
                    full_chain_records=full_chain_records,
                    best_predictions=best_predictions,
                )
            except Exception as exc:
                results[sample_idx] = {
                    'success': False,
                    'wall_time_s': time.perf_counter() - t0,
                    'returncode': -999,
                    'output_pdb': None,
                    'stdout': '',
                    'stderr': repr(exc),
                }
                continue

            results[sample_idx] = _benchmark_success_result(output_pdb, t0)

    for k in k_values:
        if k not in results:
            results[k] = {
                'success': False,
                'wall_time_s': time.perf_counter() - t0,
                'returncode': -999,
                'output_pdb': None,
                'stdout': '',
                'stderr': 'no successful best-of-k snapshot',
            }

    return results


def run_base2backbone_best_of_k_inference_batch(
    input_paths: list[str | Path],
    output_root: str | Path,
    *,
    sampler: BackboneSampler,
    device: str,
    window_size: int,
    best_of_k_list: list[int],
    window_batch_size: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> dict[str, dict[int, dict[str, Any]]]:
    output_root = Path(output_root).resolve()
    results: dict[str, dict[int, dict[str, Any]]] = {}
    k_values = sorted({int(k) for k in best_of_k_list if int(k) > 0})
    if not k_values:
        raise ValueError('best_of_k_list must contain at least one positive k')
    k_max = max(k_values)
    k_snapshot_set = set(k_values)
    pending: list[tuple[Path, Any, Any, float]] = []
    resolved_input_paths = [Path(path).resolve() for path in input_paths]

    path_iter = resolved_input_paths
    if show_progress:
        path_iter = tqdm(
            resolved_input_paths,
            desc='best-of-k prepare batch',
            leave=False,
            colour=PROGRESS_BAR_COLOR,
        )
    for input_path in path_iter:
        if show_progress and hasattr(path_iter, 'set_postfix_str'):
            path_iter.set_postfix_str(input_path.stem)
        input_key = str(input_path)
        t0 = time.perf_counter()

        if resume and _best_of_k_outputs_complete(input_path, output_root, k_values):
            results[input_key] = {
                k: _benchmark_success_result(
                    _best_of_k_output_pdb(output_root, k, input_path),
                    t0,
                    wall_time_s=0.0,
                )
                for k in k_values
            }
            continue

        try:
            _, chain_records = parse_dna(
                str(input_path),
                use_full_nucleotide=False,
                window_size=window_size,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', PDBConstructionWarning)
                _, full_chain_records = parse_dna(
                    str(input_path),
                    use_full_nucleotide=True,
                    window_size=window_size,
                )
        except Exception as exc:
            failure = {
                'success': False,
                'wall_time_s': time.perf_counter() - t0,
                'returncode': -999,
                'output_pdb': None,
                'stdout': '',
                'stderr': repr(exc),
            }
            results[input_key] = {k: failure for k in k_values}
            continue

        pending.append((input_path, chain_records, full_chain_records, t0))

    if not pending:
        return results

    best_scores: list[float | None] = [None] * len(pending)
    best_predictions: list[Any | None] = [None] * len(pending)
    pending_chain_records = [chain_records for _, chain_records, _, _ in pending]

    with torch.inference_mode():
        sample_desc = 'best-of-k diffusion samples'
        if pending:
            stems = ', '.join(path.stem for path, *_ in pending[:3])
            if len(pending) > 3:
                stems = f'{stems}, ...'
            sample_desc = f'best-of-k samples [{stems}]'
        sample_iter = range(1, k_max + 1)
        if show_progress:
            sample_iter = tqdm(
                sample_iter,
                total=k_max,
                desc=sample_desc,
                leave=False,
                colour=PROGRESS_BAR_COLOR,
            )
        for sample_idx in sample_iter:
            if show_progress and hasattr(sample_iter, 'set_postfix_str'):
                sample_iter.set_postfix_str(f'sample {sample_idx}/{k_max}')
            try:
                predictions_by_structure = predict_backbone_from_chain_records(
                    pending_chain_records,
                    sampler,
                    device,
                    window_batch_size=window_batch_size,
                )
            except Exception as exc:
                for input_path, _chain_records, _full_chain_records, t0 in pending:
                    failure = {
                        'success': False,
                        'wall_time_s': time.perf_counter() - t0,
                        'returncode': -999,
                        'output_pdb': None,
                        'stdout': '',
                        'stderr': repr(exc),
                    }
                    results[str(input_path)] = {k: failure for k in k_values}
                return results

            for batch_idx, (
                input_path,
                _chain_records,
                full_chain_records,
                t0,
            ) in enumerate(pending):
                predictions = predictions_by_structure[batch_idx]
                score = median_backbone_rmsd_vs_predictions(
                    input_path,
                    predictions,
                    window_size=window_size,
                )

                if score is not None and (
                    best_scores[batch_idx] is None or score < best_scores[batch_idx]
                ):
                    best_scores[batch_idx] = score
                    best_predictions[batch_idx] = predictions

                if sample_idx not in k_snapshot_set or best_predictions[batch_idx] is None:
                    continue

                structure_results = results.setdefault(str(input_path), {})
                try:
                    output_pdb = _write_best_of_k_snapshot(
                        input_path=input_path,
                        output_root=output_root,
                        k=sample_idx,
                        full_chain_records=full_chain_records,
                        best_predictions=best_predictions[batch_idx],
                    )
                except Exception as exc:
                    structure_results[sample_idx] = {
                        'success': False,
                        'wall_time_s': time.perf_counter() - t0,
                        'returncode': -999,
                        'output_pdb': None,
                        'stdout': '',
                        'stderr': repr(exc),
                    }
                    continue

                structure_results[sample_idx] = _benchmark_success_result(output_pdb, t0)

    for input_path, _chain_records, _full_chain_records, t0 in pending:
        structure_results = results.setdefault(str(input_path), {})
        for k in k_values:
            if k not in structure_results:
                structure_results[k] = {
                    'success': False,
                    'wall_time_s': time.perf_counter() - t0,
                    'returncode': -999,
                    'output_pdb': None,
                    'stdout': '',
                    'stderr': 'no successful best-of-k snapshot',
                }
    return results


def run_base2backbone_best_of_k_inference_pooled(
    input_path: str | Path,
    output_root: str | Path,
    *,
    best_of_k_list: list[int],
    window_batch_size: int | None = None,
    resume: bool = True,
) -> dict[int, dict[str, Any]]:
    if _pool_base2backbone is None:
        raise RuntimeError('base2backbone worker pool not initialized')
    ctx = _pool_base2backbone
    return run_base2backbone_best_of_k_inference(
        input_path,
        output_root,
        sampler=ctx['sampler'],
        device=ctx['device'],
        window_size=ctx['window_size'],
        best_of_k_list=best_of_k_list,
        window_batch_size=window_batch_size,
        resume=resume,
    )


def run_base2backbone_best_of_k_inference_batch_pooled(
    input_paths: list[str | Path],
    output_root: str | Path,
    *,
    best_of_k_list: list[int],
    window_batch_size: int | None = None,
    resume: bool = True,
) -> dict[str, dict[int, dict[str, Any]]]:
    if _pool_base2backbone is None:
        raise RuntimeError('base2backbone worker pool not initialized')
    ctx = _pool_base2backbone
    return run_base2backbone_best_of_k_inference_batch(
        input_paths,
        output_root,
        sampler=ctx['sampler'],
        device=ctx['device'],
        window_size=ctx['window_size'],
        best_of_k_list=best_of_k_list,
        window_batch_size=window_batch_size,
        resume=resume,
    )


def run_knn_baseline_structure(
    input_path: str | Path,
    output_dir: str | Path,
    knn_state: 'KnnBaselineState',
    *,
    window_size: int,
    resume: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / f'{input_path.stem}_knn_baseline.pdb'
    t0 = time.perf_counter()

    if resume and output_pdb.is_file() and output_pdb.stat().st_size > 0:
        return _benchmark_success_result(output_pdb, t0, wall_time_s=0.0)

    try:
        predictions = knn_state.predictions_for_pdb(input_path.stem)
        if not predictions:
            raise RuntimeError('no kNN predictions for structure')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                str(input_path),
                use_full_nucleotide=True,
                window_size=window_size,
            )
        write_structure(
            build_output_universe(full_chain_records, predictions),
            output_pdb,
        )
    except Exception as exc:
        return {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(exc),
        }

    return {
        'success': True,
        'wall_time_s': time.perf_counter() - t0,
        'returncode': 0,
        'output_pdb': output_pdb,
        'stdout': '',
        'stderr': '',
    }


