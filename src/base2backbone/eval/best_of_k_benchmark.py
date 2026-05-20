"""Best-of-k full-structure export with batched MolProbity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from .molprobity import annotate_benchmark_rows_with_molprobity
from .structure_runners import (
    CheckpointSamplerAdapter,
    run_base2backbone_best_of_k_inference_batch,
)
from ..runtime import PROGRESS_BAR_COLOR


def _best_of_k_benchmark_row(input_path: Path, res: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': input_path.stem,
        'input_path': str(input_path),
        'success': res['success'],
        'wall_time_s': res['wall_time_s'],
        'returncode': res['returncode'],
        'output_pdb': str(res['output_pdb']) if res['output_pdb'] else '',
        'stderr': res['stderr'][:1000],
    }


def run_base2backbone_best_of_k_benchmark(
    input_paths: list[str | Path],
    output_root: str | Path,
    *,
    checkpoint_model,
    device: str,
    window_size: int,
    best_of_k_list: list[int],
    num_timesteps: int | None = None,
    batch_size: int = 1,
    window_batch_size: int | None = None,
    resume: bool = True,
    molprobity_timeout_s: int = 600,
    molprobity_max_workers: int = 32,
) -> dict[int, list[dict[str, Any]]]:
    """Export best-of-k structures first, then run MolProbity in large batches."""
    k_values = sorted({int(k) for k in best_of_k_list if int(k) > 0})
    rows_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in k_values}
    resolved_paths = [Path(path).resolve() for path in input_paths]
    sampler = CheckpointSamplerAdapter(checkpoint_model, num_timesteps)
    n_structures = len(resolved_paths)
    n_batches = (n_structures + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(tqdm(
        range(0, n_structures, batch_size),
        total=n_batches,
        desc=f'best-of-k export ({n_structures} structures)',
        leave=False,
        colour=PROGRESS_BAR_COLOR,
    )):
        batch_paths = resolved_paths[batch_start:batch_start + batch_size]
        batch_results = run_base2backbone_best_of_k_inference_batch(
            batch_paths,
            output_root,
            sampler=sampler,
            device=device,
            window_size=window_size,
            best_of_k_list=best_of_k_list,
            window_batch_size=window_batch_size,
            resume=resume,
            show_progress=True,
        )
        for input_path in batch_paths:
            per_k = batch_results[str(input_path)]
            for k in k_values:
                rows_by_k[k].append(_best_of_k_benchmark_row(input_path, per_k[k]))

    for k in k_values:
        annotate_benchmark_rows_with_molprobity(
            rows_by_k[k],
            timeout_s=molprobity_timeout_s,
            max_workers=molprobity_max_workers,
            resume=resume,
        )
    return rows_by_k
