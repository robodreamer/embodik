"""Global scoring functions for reachability analysis.

This module provides functions to compute global scores from reachability maps,
supporting both FK-based (forward sampling) and IK-based (inverse sampling) approaches.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

LOGGER = logging.getLogger("embodik.reachability_scoring")

__all__ = [
    "compute_average_metric_score",
    "compute_weighted_metric_score",
    "compute_coverage_score",
    "compute_composite_score",
    "compute_fk_global_scores",
    "compute_ik_global_scores",
    "compare_fk_ik_scores",
]


def compute_average_metric_score(
    metric_values: np.ndarray,
    metric_name: str = "Manipulability",
) -> Dict[str, float]:
    """Compute average metric score across all valid points.

    Matches MATLAB: GM = sum(metrics) / count
    Ideal value: 1.0 (all points reachable with max metric)

    Args:
        metric_values: Array of metric values
        metric_name: Name of the metric (for logging)

    Returns:
        Dictionary with:
        - "average": Average metric value
        - "sum": Sum of all metric values
        - "count": Number of valid points
        - "min": Minimum metric value
        - "max": Maximum metric value
        - "std": Standard deviation
    """
    if len(metric_values) == 0:
        return {
            "average": 0.0,
            "sum": 0.0,
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    return {
        "average": float(np.mean(metric_values)),
        "sum": float(np.sum(metric_values)),
        "count": int(len(metric_values)),
        "min": float(np.min(metric_values)),
        "max": float(np.max(metric_values)),
        "std": float(np.std(metric_values)),
    }


def compute_weighted_metric_score(
    metric_values: np.ndarray,
    weights: np.ndarray,
    metric_name: str = "Manipulability",
) -> Dict[str, float]:
    """Compute visitation-weighted average metric score.

    For FK approach: weights = visitation counts
    Gives more weight to frequently visited regions.

    Args:
        metric_values: Array of metric values
        weights: Array of weights (e.g., visitation counts)
        metric_name: Name of the metric (for logging)

    Returns:
        Dictionary with all fields from compute_average_metric_score, plus:
        - "weighted_average": Weighted average metric value
        - "total_weight": Sum of all weights
    """
    if len(metric_values) == 0 or len(weights) == 0:
        return compute_average_metric_score(metric_values, metric_name)

    if len(metric_values) != len(weights):
        LOGGER.warning(
            "Metric values and weights have different lengths (%d vs %d), using simple average",
            len(metric_values),
            len(weights),
        )
        return compute_average_metric_score(metric_values, metric_name)

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return compute_average_metric_score(metric_values, metric_name)

    weighted_sum = float(np.sum(metric_values * weights))
    weighted_avg = weighted_sum / total_weight

    result = compute_average_metric_score(metric_values, metric_name)
    result["weighted_average"] = weighted_avg
    result["total_weight"] = total_weight
    return result


def compute_coverage_score(
    valid_count: int,
    total_count: int,
    metric_values: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute workspace coverage score.

    For IK approach: what percentage of grid points are reachable.

    Args:
        valid_count: Number of valid (reachable) points
        total_count: Total number of grid points
        metric_values: Optional array of metric values for valid points

    Returns:
        Dictionary with:
        - "coverage_ratio": valid_count / total_count
        - "valid_count": Number of valid points
        - "invalid_count": Number of invalid points
        - "total_count": Total number of points
        - "metric_average": Average metric for valid points (if provided)
    """
    if total_count == 0:
        return {
            "coverage_ratio": 0.0,
            "valid_count": 0,
            "invalid_count": 0,
            "total_count": 0,
            "metric_average": 0.0,
        }

    coverage = float(valid_count) / float(total_count)
    metric_avg = (
        float(np.mean(metric_values))
        if metric_values is not None and len(metric_values) > 0
        else 0.0
    )

    return {
        "coverage_ratio": coverage,
        "valid_count": valid_count,
        "invalid_count": total_count - valid_count,
        "total_count": total_count,
        "metric_average": metric_avg,
    }


def compute_composite_score(
    metrics_dict: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute composite score from multiple metrics.

    Args:
        metrics_dict: Dictionary mapping metric names to arrays of values
        weights: Optional weights for each metric (default: equal weights)

    Returns:
        Dictionary with:
        - "composite_average": Weighted average across all metrics
        - "individual_scores": Dictionary of per-metric averages
        - "weights": Dictionary of applied weights
    """
    if not metrics_dict:
        return {
            "composite_average": 0.0,
            "individual_scores": {},
            "weights": {},
        }

    if weights is None:
        weights = {name: 1.0 / len(metrics_dict) for name in metrics_dict.keys()}

    individual_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, metric_values in metrics_dict.items():
        avg = float(np.mean(metric_values)) if len(metric_values) > 0 else 0.0
        individual_scores[metric_name] = avg
        weight = weights.get(metric_name, 0.0)
        weighted_sum += avg * weight
        total_weight += weight

    composite = weighted_sum / total_weight if total_weight > 0.0 else 0.0

    return {
        "composite_average": composite,
        "individual_scores": individual_scores,
        "weights": weights,
    }


def compute_fk_global_scores(
    reach_map: np.ndarray,
    metric_column_indices: Dict[str, int],
    visitation_column: int = 6,
) -> Dict[str, Dict[str, float]]:
    """Compute global scores from FK reachability map.

    Args:
        reach_map: Binned reachability map (num_voxels, num_columns)
        metric_column_indices: Mapping of metric names to column indices
        visitation_column: Column index for visitation counts

    Returns:
        Dictionary mapping metric names to score dictionaries
    """
    if torch is not None and isinstance(reach_map, torch.Tensor):
        reach_map_np = reach_map.cpu().numpy()
    else:
        reach_map_np = np.asarray(reach_map)

    if reach_map_np.shape[0] == 0:
        return {}

    metrics_dict = {}

    for metric_name, col_idx in metric_column_indices.items():
        if col_idx >= reach_map_np.shape[1]:
            LOGGER.warning(
                "Metric column index %d out of range for %s (shape: %s)",
                col_idx,
                metric_name,
                reach_map_np.shape,
            )
            continue

        metric_values = reach_map_np[:, col_idx]
        visitation = (
            reach_map_np[:, visitation_column]
            if visitation_column < reach_map_np.shape[1]
            else np.ones(len(metric_values))
        )

        # Only consider visited voxels
        visited_mask = visitation > 0
        if np.any(visited_mask):
            visited_metrics = metric_values[visited_mask]
            visited_weights = visitation[visited_mask]

            # Compute both simple and weighted averages
            simple_score = compute_average_metric_score(visited_metrics, metric_name)
            weighted_score = compute_weighted_metric_score(
                visited_metrics, visited_weights, metric_name
            )

            metrics_dict[metric_name] = {
                **simple_score,
                **weighted_score,
            }
        else:
            metrics_dict[metric_name] = compute_average_metric_score(
                np.array([]), metric_name
            )

    return metrics_dict


def compute_ik_global_scores(
    valid_points: np.ndarray,
    invalid_points: np.ndarray,
    valid_metrics: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute global scores from IK reachability analysis.

    Args:
        valid_points: Array of valid (reachable) points
        invalid_points: Array of invalid (unreachable) points
        valid_metrics: Dictionary mapping metric names to arrays of values for valid points

    Returns:
        Dictionary with:
        - "coverage": Coverage score dictionary
        - "metrics": Dictionary mapping metric names to score dictionaries
    """
    total_count = len(valid_points) + len(invalid_points)
    valid_count = len(valid_points)

    # Compute coverage score
    coverage_score = compute_coverage_score(valid_count, total_count)

    # Compute metric scores for valid points
    metric_scores = {}
    for metric_name, metric_values in valid_metrics.items():
        if len(metric_values) != len(valid_points):
            LOGGER.warning(
                "Metric %s has %d values but %d valid points, skipping",
                metric_name,
                len(metric_values),
                len(valid_points),
            )
            continue

        metric_scores[metric_name] = compute_average_metric_score(
            metric_values, metric_name
        )
        # Add coverage-weighted metric (penalize unreachable regions)
        coverage_weighted = (
            metric_scores[metric_name]["average"] * coverage_score["coverage_ratio"]
        )
        metric_scores[metric_name]["coverage_weighted_average"] = coverage_weighted

    return {
        "coverage": coverage_score,
        "metrics": metric_scores,
    }


def compare_fk_ik_scores(
    fk_scores: Dict[str, Dict[str, float]],
    ik_scores: Dict[str, Dict[str, float]],
) -> Dict[str, any]:
    """Compare global scores between FK and IK approaches.

    Args:
        fk_scores: Global scores from FK approach
        ik_scores: Global scores from IK approach

    Returns:
        Comparison dictionary with:
        - "fk_scores": FK scores
        - "ik_scores": IK scores
        - "differences": Dictionary of metric differences
        - "insights": List of insight strings
    """
    comparison = {
        "fk_scores": fk_scores,
        "ik_scores": ik_scores,
        "differences": {},
        "insights": [],
    }

    # Compare metric averages
    fk_metrics = fk_scores if isinstance(fk_scores, dict) else {}
    ik_metrics = ik_scores.get("metrics", {}) if isinstance(ik_scores, dict) else {}

    all_metric_names = set(fk_metrics.keys()) | set(ik_metrics.keys())

    for metric_name in all_metric_names:
        fk_avg = fk_metrics.get(metric_name, {}).get("average", 0.0)
        ik_avg = ik_metrics.get(metric_name, {}).get("average", 0.0)

        diff = ik_avg - fk_avg
        comparison["differences"][metric_name] = {
            "fk_average": fk_avg,
            "ik_average": ik_avg,
            "difference": diff,
            "relative_difference": diff / max(fk_avg, 1e-6) if fk_avg > 0 else 0.0,
        }

        if abs(diff) > 0.1:
            comparison["insights"].append(
                f"{metric_name}: IK average ({ik_avg:.3f}) differs significantly from FK ({fk_avg:.3f})"
            )

    # Coverage comparison (IK-specific)
    if "coverage" in ik_scores:
        ik_coverage = ik_scores["coverage"]["coverage_ratio"]
        # Estimate FK coverage from visitation
        fk_visited = sum(
            scores.get("count", 0) for scores in fk_metrics.values()
        )
        comparison["coverage_comparison"] = {
            "ik_coverage": ik_coverage,
            "fk_visited_voxels": fk_visited,
            "note": "FK coverage estimation requires total voxel count",
        }

    return comparison

