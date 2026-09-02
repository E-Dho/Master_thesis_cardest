from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

from model.src.data.full_join_sampler import FullJoinBatch, NeuroCardFullJoinSampleSource
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.trajectory_distinct import (
    CompactTrajectorySegmentIndex,
    ONE_PASS_AR_SEGMENT_MEASURE_CAPABILITY,
    PhysicalSpatialPredicate,
    SegmentSpatialPredicate,
    SegmentTemporalPredicate,
    TRAJECTORY_TARGET_SEMANTICS_VERSION,
    TrajectoryDistinctNotApplicable,
    TrajectoryDistinctRuntimeConfig,
    TrajectorySegmentIndex,
    TrajectoryQuerySemantics,
    UnsupportedTrajectoryContext,
    context_satisfies_row_with_trajectory_semantics,
    segment_rectangle_intersects_mask,
    temporal_overlap_mask,
    trajectory_base_measure_support,
    trajectory_distinct_context_eligibility,
    write_compact_trajectory_index,
)
from model.src.evaluation.exact_evaluator import ExactOracle
from model.src.evaluation.pol_query_adapter import (
    assert_checkpoint_trajectory_config_compatible,
    evaluate_pol_distinct_record,
    pol_workload_record_to_context,
)
from model.src.predicates.generation import GeneratedTrainingContext, PredicateTrainingContextGenerator
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.vocabulary import PredicateVocabularies
from model.src.training.losses import (
    effective_sample_size,
    stable_combine_importance_and_terminal_inverse_weights,
    terminal_inverse_fanout_weights,
    terminal_log_weights,
)

if HAS_TORCH:
    from model.src.inference.estimator import OnePassEstimator
    from model.src.inference.torch_estimator import TorchDistributionModel
    from model.src.model.checkpoint import load_resmade_checkpoint, save_resmade_checkpoint
    from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
    from model.src.model.resmade import TrajectoryDistinctConfig
    from model.src.training.resmade_trainer import (
        _RunningTrajectoryDistinctStats,
        _run_validation,
        _validation_sample_source_from_config,
        train_resmade_sample_source,
        trajectory_dedup_loss_for_batch,
    )


def _trajectory_metadata() -> ModelMetadata:
    columns = (
        ColumnMetadata("segment.x", ColumnKind.DATA, (0, 1), table="segments"),
        ColumnMetadata("segment.y", ColumnKind.DATA, (0, 1), table="segments"),
        ColumnMetadata("trip.kind", ColumnKind.DATA, ("train", "test"), table="trips"),
        ColumnMetadata("I_trips", ColumnKind.INDICATOR, (0, 1), table="trips"),
        ColumnMetadata("I_segments", ColumnKind.INDICATOR, (0, 1), table="segments"),
        ColumnMetadata(
            "F_trips_to_segments",
            ColumnKind.FANOUT,
            (1, 2, 10),
            table="segments",
            fanout_source="trips->segments",
        ),
    )
    return ModelMetadata(
        columns=columns,
        full_join_cardinality=6,
        join_root="trips",
        join_tables=("trips", "segments"),
        join_edges=(("trips", "segments"),),
    )


def _trajectory_rows(metadata: ModelMetadata) -> tuple[np.ndarray, tuple[str, ...]]:
    decoded = (
        (1, 0, "train", 1, 1, 2),  # A matches Q
        (0, 0, "train", 1, 1, 2),  # A does not match Q
        (1, 0, "train", 1, 1, 10),  # B matches Q
        (1, 1, "train", 1, 1, 10),  # B matches Q
        (0, 1, "train", 1, 1, 10),  # B does not match Q
        (0, 0, "test", 1, 1, 1),  # C does not match Q
    )
    trajectory_ids = ("A", "A", "B", "B", "B", "C")
    encoded = np.zeros((len(decoded), len(metadata.columns)), dtype=np.int64)
    for row_index, row in enumerate(decoded):
        for column_index, value in enumerate(row):
            encoded[row_index, column_index] = metadata.columns[column_index].encode_value(value)
    return encoded, trajectory_ids


def _pol_segment_metadata() -> ModelMetadata:
    columns = (
        ColumnMetadata("segments:segment_idx", ColumnKind.DATA, (0, 1, 2), table="segments"),
        ColumnMetadata("segments:t_s", ColumnKind.DATA, (0.0, 10.0, 20.0), table="segments"),
        ColumnMetadata("segments:t_e", ColumnKind.DATA, (10.0, 20.0, 30.0), table="segments"),
        ColumnMetadata("segments:s_x", ColumnKind.DATA, (-1.0, 0.0, 1.0, 3.0), table="segments"),
        ColumnMetadata("segments:s_y", ColumnKind.DATA, (0.0, 1.0, 3.0), table="segments"),
        ColumnMetadata("segments:e_x", ColumnKind.DATA, (1.0, 2.0, 3.0, 4.0), table="segments"),
        ColumnMetadata("segments:e_y", ColumnKind.DATA, (1.0, 2.0, 3.0, 4.0), table="segments"),
        ColumnMetadata("I_trips", ColumnKind.INDICATOR, (0, 1), table="trips"),
        ColumnMetadata("I_segments", ColumnKind.INDICATOR, (0, 1), table="segments"),
    )
    return ModelMetadata(
        columns=columns,
        full_join_cardinality=3,
        join_root="trips",
        join_tables=("trips", "segments"),
        join_edges=(("trips", "segments"),),
    )


def _encode_pol_segments(
    metadata: ModelMetadata,
    decoded: tuple[tuple[object, ...], ...],
) -> np.ndarray:
    rows = np.zeros((len(decoded), len(metadata.columns)), dtype=np.int64)
    for row_index, row in enumerate(decoded):
        for column_index, value in enumerate(row):
            rows[row_index, column_index] = metadata.columns[column_index].encode_value(value)
    return rows


def _query_context() -> GeneratedTrainingContext:
    tokens = (
        PredicateToken.equal(1),
        PredicateToken.wildcard(),
        PredicateToken.wildcard(),
        PredicateToken.equal(1),
        PredicateToken.equal(1),
        PredicateToken.wildcard(),
    )
    return GeneratedTrainingContext(
        tokens=tokens,
        included_tables=frozenset({"trips", "segments"}),
        inverse_fanout_columns=frozenset(),
        ordinary_predicates={"segment.x": tokens[0]},
    )


def _write_prepared_pol_fixture(
    root: Path,
    *,
    metadata: ModelMetadata,
    rows: np.ndarray,
    trajectory_ids: tuple[int, ...],
    segment_ids: tuple[tuple[int, int], ...],
) -> None:
    manifest = {
        "dataset_name": root.name,
        "dataset_type": "pol_trajectory_full_join",
        "join_cardinality": int(rows.shape[0]),
        "metadata": metadata.to_json_dict(),
        "domains_complete": True,
        "metadata_source": "complete_base_tables_and_join_metadata",
        "sample_rows": int(rows.shape[0]),
        "format_version": 2,
    }
    root.mkdir(parents=True, exist_ok=True)
    Path(root, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(root / "sample_rows.npy", rows)
    np.save(root / "sample_trajectory_ids.npy", np.asarray(trajectory_ids, dtype=np.int64))
    np.save(root / "sample_segment_ids.npy", np.asarray(segment_ids, dtype=np.int64))
    write_compact_trajectory_index(
        root / "trajectory_segment_index",
        metadata=metadata,
        trip_ids=trajectory_ids,
        segment_idx=tuple(segment_id[1] for segment_id in segment_ids),
        t_s=tuple(float(metadata.columns[1].domain[int(row[1])]) for row in rows),
        t_e=tuple(float(metadata.columns[2].domain[int(row[2])]) for row in rows),
        s_x=tuple(float(metadata.columns[3].domain[int(row[3])]) for row in rows),
        s_y=tuple(float(metadata.columns[4].domain[int(row[4])]) for row in rows),
        e_x=tuple(float(metadata.columns[5].domain[int(row[5])]) for row in rows),
        e_y=tuple(float(metadata.columns[6].domain[int(row[6])]) for row in rows),
        segment_varying_columns=(
            "segments:segment_idx",
            "segments:t_s",
            "segments:t_e",
            "segments:s_x",
            "segments:s_y",
            "segments:e_x",
            "segments:e_y",
        ),
    )


class _TrajectorySource:
    def __init__(
        self,
        metadata: ModelMetadata,
        rows: np.ndarray,
        trajectory_ids: tuple[str, ...],
        provider: TrajectorySegmentIndex,
    ) -> None:
        self.metadata = metadata
        self.rows = rows
        self.trajectory_ids = trajectory_ids
        self.trajectory_multiplicity_provider = provider

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(self.rows), size=batch_size)
        return FullJoinBatch(
            encoded_values=self.rows[indices],
            column_metadata=self.metadata.columns,
            trajectory_ids=tuple(self.trajectory_ids[int(index)] for index in indices),
            fixture_rows_reused=batch_size,
        )


class _FakeDistinctEstimator:
    def __init__(self, *, matching: float = 80.0, dedup: float = 0.5) -> None:
        self.matching = matching
        self.dedup = dedup

    def estimate_distinct_trajectories(self, *args, **kwargs):
        del args, kwargs
        return type(
            "FakeDistinctEstimate",
            (),
            {
                "matching_segment_estimate": self.matching,
                "traj_dedup_factor": self.dedup,
                "distinct_trajectory_estimate": self.matching * self.dedup,
                "model_forward_calls": 1,
            },
        )()


class TrajectoryDistinctTest(unittest.TestCase):
    def test_synthetic_identity_m_times_expected_inverse_m_equals_distinct(self) -> None:
        metadata = _trajectory_metadata()
        rows, trajectory_ids = _trajectory_rows(metadata)
        provider = TrajectorySegmentIndex.from_rows(
            metadata=metadata,
            trajectory_ids=trajectory_ids,
            encoded_rows=rows,
            predicate_columns=("segment.x",),
            trajectory_key="trip_id",
        )
        base_context = _query_context()
        tokens = list(base_context.tokens)
        tokens[-1] = PredicateToken.inv_fanout()
        context = GeneratedTrainingContext(
            tokens=tuple(tokens),
            included_tables=base_context.included_tables,
            inverse_fanout_columns=frozenset({"F_trips_to_segments"}),
            ordinary_predicates=base_context.ordinary_predicates,
        )
        multiplicities = provider.matching_multiplicities(
            anchor_trajectory_ids=("A", "B", "B"),
            contexts=(context, context, context),
        )
        self.assertEqual(tuple(multiplicities), (1, 2, 2))
        inverse_mean = float(np.mean(1.0 / multiplicities))
        self.assertAlmostEqual(3.0 * inverse_mean, 2.0)

    def test_provider_targets_and_unsupported_predicates(self) -> None:
        metadata = _trajectory_metadata()
        rows, trajectory_ids = _trajectory_rows(metadata)
        provider = TrajectorySegmentIndex.from_rows(
            metadata=metadata,
            trajectory_ids=trajectory_ids,
            encoded_rows=rows,
            predicate_columns=("segment.x",),
            trajectory_key="trip_id",
        )
        context = _query_context()
        targets = provider.local_targets(
            anchor_trajectory_ids=("A", "B"),
            contexts=(context, context),
        )
        self.assertTrue(np.allclose(targets, [1.0, 0.5]))
        unsupported = GeneratedTrainingContext(
            tokens=(
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
                PredicateToken.equal("train"),
                PredicateToken.equal(1),
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
            ),
            included_tables=frozenset({"trips", "segments"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={"trip.kind": PredicateToken.equal("train")},
        )
        with self.assertRaises(UnsupportedTrajectoryContext):
            provider.matching_multiplicities(
                anchor_trajectory_ids=("A",),
                contexts=(unsupported,),
            )

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_terminal_masks_can_use_final_fanout_without_earlier_leakage(self) -> None:
        metadata = _trajectory_metadata()
        vocab = PredicateVocabularies.from_metadata(metadata)
        model = PredicateResMADE(
            PredicateResMADEConfig(
                predicate_input_bins=vocab.input_bins,
                data_output_bins=metadata.data_output_bins,
                column_kinds=tuple(column.kind.value for column in metadata.columns),
                hidden_sizes=(24, 24),
                direct_io_connections=True,
                direct_io_source_kinds=("data", "indicator", "fanout"),
                direct_io_destination_kinds=("data", "indicator", "fanout"),
                embedding_size=4,
                trajectory_distinct_config=TrajectoryDistinctConfig(enabled=True),
            )
        )
        assert model.traj_dedup_direct_io_layer is not None
        assert model.direct_io_layer is not None
        starts = np.cumsum((0, *model.column_input_widths[:-1]))
        final_fanout_index = len(metadata.columns) - 1
        fanout_start = int(starts[final_fanout_index])
        fanout_stop = fanout_start + model.column_input_widths[final_fanout_index]
        self.assertGreater(
            float(model.traj_dedup_direct_io_layer.mask[:, fanout_start:fanout_stop].sum()),
            0.0,
        )
        first_out_start, first_out_stop = model.output_slices[0]
        self.assertEqual(
            float(model.direct_io_layer.mask[first_out_start:first_out_stop, fanout_start:fanout_stop].sum()),
            0.0,
        )
        self.assertEqual(model.traj_dedup_head.mask.shape[0], 1)
        self.assertEqual(len(model.config.predicate_input_bins), len(metadata.columns))

    def test_terminal_inverse_weights_include_all_active_fanouts(self) -> None:
        metadata = _trajectory_metadata()
        rows, _ = _trajectory_rows(metadata)
        contexts = [
            [
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ],
            [
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ],
        ]
        selected_rows = rows[[0, 2]]
        inv = terminal_inverse_fanout_weights(selected_rows, contexts, metadata)
        self.assertTrue(np.allclose(inv, [0.5, 0.1]))
        combined = stable_combine_importance_and_terminal_inverse_weights(
            inv,
            np.asarray([2.0, 5.0]),
        )
        reference = np.asarray([1.0, 0.5])
        self.assertTrue(np.allclose(combined, reference))

    def test_terminal_log_weights_preserve_unscaled_importance_weight(self) -> None:
        metadata = _trajectory_metadata()
        rows, _ = _trajectory_rows(metadata)
        contexts = [
            [
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ],
            [
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ],
        ]
        selected_rows = rows[[0, 2]]
        log_weights = terminal_log_weights(
            selected_rows,
            contexts,
            metadata,
            rho=np.asarray([1000.0, 5.0]),
        )
        self.assertTrue(np.allclose(np.exp(log_weights), [500.0, 0.5]))

    def test_trip_only_query_is_not_segment_measure(self) -> None:
        metadata = _trajectory_metadata()
        rows, trajectory_ids = _trajectory_rows(metadata)
        provider = TrajectorySegmentIndex.from_rows(
            metadata=metadata,
            trajectory_ids=trajectory_ids,
            encoded_rows=rows,
            trajectory_key="trip_id",
            segment_varying_columns=("segment.x",),
            segment_table="segments",
        )
        trip_only = GeneratedTrainingContext(
            tokens=(
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.equal("train"),
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ),
            included_tables=frozenset({"trips"}),
            inverse_fanout_columns=frozenset({"F_trips_to_segments"}),
            ordinary_predicates={"trip.kind": PredicateToken.equal("train")},
        )
        eligibility = trajectory_distinct_context_eligibility(
            trip_only,
            metadata,
            provider.runtime_config,
        )
        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.reason, "non_segment_measure")
        result = provider.evaluate_batch(
            anchor_trajectory_ids=("B",),
            contexts=(trip_only,),
        )
        self.assertFalse(bool(result.eligible_mask[0]))
        self.assertEqual(result.skip_reasons[0], "non_segment_measure")

    def test_static_predicate_does_not_reduce_local_segment_multiplicity(self) -> None:
        metadata = _trajectory_metadata()
        rows, trajectory_ids = _trajectory_rows(metadata)
        provider = TrajectorySegmentIndex.from_rows(
            metadata=metadata,
            trajectory_ids=trajectory_ids,
            encoded_rows=rows,
            trajectory_key="trip_id",
            trajectory_static_columns=("trip.kind",),
            segment_varying_columns=("segment.x",),
            segment_table="segments",
        )
        static_only = GeneratedTrainingContext(
            tokens=(
                PredicateToken.wildcard(),
                PredicateToken.wildcard(),
                PredicateToken.equal("train"),
                PredicateToken.equal(1),
                PredicateToken.equal(1),
                PredicateToken.wildcard(),
            ),
            included_tables=frozenset({"trips", "segments"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={"trip.kind": PredicateToken.equal("train")},
        )
        m = provider.matching_multiplicities(
            anchor_trajectory_ids=("B",),
            contexts=(static_only,),
        )
        self.assertEqual(int(m[0]), 3)
        varying = _query_context()
        m2 = provider.matching_multiplicities(
            anchor_trajectory_ids=("B",),
            contexts=(varying,),
        )
        self.assertEqual(int(m2[0]), 2)

    def test_temporal_overlap_boundaries_match_pol_sql(self) -> None:
        starts = np.asarray([0, 10, 20, 5, 12], dtype=float)
        ends = np.asarray([10, 20, 30, 25, 13], dtype=float)
        mask = temporal_overlap_mask(starts, ends, lower=10, upper=20)
        # start < upper and end >= lower: ending exactly at lower matches;
        # starting exactly at upper does not.
        self.assertTrue(np.array_equal(mask, [True, True, False, True, True]))

    def test_spatial_segment_rectangle_intersections_include_boundary_touches(self) -> None:
        sx = np.asarray([1, -1, -1, 0, 2, -1, 0], dtype=float)
        sy = np.asarray([1, -1, 1, 0, 2, 0, 2], dtype=float)
        ex = np.asarray([2, -2, 3, 0, 3, 0, 2], dtype=float)
        ey = np.asarray([2, -2, 1, 3, 3, 0, 0], dtype=float)
        mask = segment_rectangle_intersects_mask(sx, sy, ex, ey, 0, 0, 2, 2)
        self.assertTrue(
            np.array_equal(mask, [True, False, True, True, True, True, True])
        )

    def test_crossing_segment_proves_endpoint_ranges_not_equivalent_to_intersects(self) -> None:
        intersects = segment_rectangle_intersects_mask(
            np.asarray([-1.0]),
            np.asarray([1.0]),
            np.asarray([3.0]),
            np.asarray([1.0]),
            0.0,
            0.0,
            2.0,
            2.0,
        )
        endpoint_containment = (
            0.0 <= -1.0 <= 2.0
            and 0.0 <= 3.0 <= 2.0
            and 0.0 <= 1.0 <= 2.0
            and 0.0 <= 1.0 <= 2.0
        )
        self.assertTrue(bool(intersects[0]))
        self.assertFalse(endpoint_containment)

    def test_spatial_base_measure_is_intentionally_unsupported(self) -> None:
        metadata = _pol_segment_metadata()
        context = GeneratedTrainingContext(
            tokens=tuple(PredicateToken.wildcard() for _ in metadata.columns),
            included_tables=frozenset({"trips", "segments"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={},
            trajectory_query=TrajectoryQuerySemantics(
                spatial_predicates=(SegmentSpatialPredicate(0.0, 0.0, 2.0, 2.0),),
            ),
        )
        self.assertFalse(ONE_PASS_AR_SEGMENT_MEASURE_CAPABILITY.supports_spatial_intersects)
        support = trajectory_base_measure_support(context)
        self.assertFalse(support.eligible)
        self.assertEqual(support.reason, "unsupported_base_segment_spatial_measure")

    def test_temporal_trajectory_target_remains_supported(self) -> None:
        metadata = _pol_segment_metadata()
        context = GeneratedTrainingContext(
            tokens=tuple(PredicateToken.wildcard() for _ in metadata.columns),
            included_tables=frozenset({"trips", "segments"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={},
            trajectory_query=TrajectoryQuerySemantics(
                temporal_predicates=(
                    SegmentTemporalPredicate("segments:t_s", "segments:t_e", lower=10.0, upper=20.0),
                ),
            ),
        )
        self.assertTrue(trajectory_base_measure_support(context).eligible)

    def test_exact_oracle_counts_logical_segment_ids_not_duplicate_rows(self) -> None:
        metadata = _trajectory_metadata()
        rows, _ = _trajectory_rows(metadata)
        duplicated = np.concatenate([rows[[0]], rows[[0]], rows[[2]]], axis=0)
        oracle = ExactOracle(metadata, duplicated)
        context = _query_context()
        result = oracle.exact_distinct_trajectory_count(
            context,
            trajectory_ids=("A", "A", "B"),
            segment_ids=(("A", 0), ("A", 0), ("B", 0)),
        )
        self.assertEqual(result.matching_segments_true, 2)
        self.assertEqual(result.distinct_trajectories_true, 2)
        self.assertAlmostEqual(result.a_true, 1.0)

    def test_compact_mmap_index_evaluates_pol_temporal_spatial_semantics(self) -> None:
        metadata = _pol_segment_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_compact_trajectory_index(
                tmpdir,
                metadata=metadata,
                trip_ids=(10, 10, 20),
                segment_idx=(0, 1, 0),
                t_s=(0.0, 10.0, 20.0),
                t_e=(10.0, 20.0, 30.0),
                s_x=(0.0, 3.0, 0.0),
                s_y=(0.0, 3.0, 3.0),
                e_x=(1.0, 4.0, 2.0),
                e_y=(1.0, 4.0, 3.0),
                trajectory_static_columns=(),
                segment_varying_columns=(
                    "segments:segment_idx",
                    "segments:t_s",
                    "segments:t_e",
                    "segments:s_x",
                    "segments:s_y",
                    "segments:e_x",
                    "segments:e_y",
                ),
                srid=26916,
            )
            provider = CompactTrajectorySegmentIndex.from_directory(tmpdir)
            self.assertEqual(manifest["index_type"], "compact_pol_segment_mmap")
            self.assertEqual(provider.segment_count, 3)
            self.assertGreater(provider.storage_summary()["estimated_bytes_50m_segments"], 0)
            self.assertTrue(isinstance(provider.segment_idx, np.memmap))
            context = GeneratedTrainingContext(
                tokens=(
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.equal(1),
                    PredicateToken.equal(1),
                ),
                included_tables=frozenset({"trips", "segments"}),
                inverse_fanout_columns=frozenset(),
                ordinary_predicates={},
                trajectory_query=TrajectoryQuerySemantics(
                    temporal_predicates=(
                        SegmentTemporalPredicate(
                            "segments:t_s",
                            "segments:t_e",
                            lower=10.0,
                            upper=20.0,
                        ),
                    ),
                    spatial_predicates=(
                        SegmentSpatialPredicate(
                            0.0,
                            0.0,
                            2.0,
                            2.0,
                            srid=26916,
                        ),
                    ),
                ),
            )
            result = provider.evaluate_batch(
                anchor_trajectory_ids=(10,),
                contexts=(context,),
            )
            self.assertTrue(bool(result.eligible_mask[0]))
            self.assertEqual(int(result.multiplicities[0]), 1)

    def test_compact_index_rejects_non_segment_measure(self) -> None:
        metadata = _pol_segment_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_compact_trajectory_index(
                tmpdir,
                metadata=metadata,
                trip_ids=(10,),
                segment_idx=(0,),
                t_s=(0.0,),
                t_e=(10.0,),
                s_x=(0.0,),
                s_y=(0.0,),
                e_x=(1.0,),
                e_y=(1.0,),
                segment_varying_columns=("segments:segment_idx",),
            )
            provider = CompactTrajectorySegmentIndex.from_directory(tmpdir)
            context = GeneratedTrainingContext(
                tokens=tuple(PredicateToken.wildcard() for _ in metadata.columns),
                included_tables=frozenset({"trips"}),
                inverse_fanout_columns=frozenset(),
                ordinary_predicates={},
            )
            result = provider.evaluate_batch(
                anchor_trajectory_ids=(10,),
                contexts=(context,),
            )
            self.assertFalse(bool(result.eligible_mask[0]))
            self.assertEqual(result.skip_reasons[0], "non_segment_measure")

    def test_generated_pol_context_populates_trajectory_query(self) -> None:
        metadata = _pol_segment_metadata()
        row = _encode_pol_segments(
            metadata,
            ((1, 10.0, 20.0, 1.0, 1.0, 2.0, 2.0, 1, 1),),
        )
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "row_satisfied",
                "table_subset_sampling": "full",
                "wildcard_probability": 1.0,
                "equality_probability": 0.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "trajectory_query_semantics": "pol_segments",
                "trajectory_temporal_probability": 1.0,
                "trajectory_spatial_probability": 1.0,
                "trajectory_srid": 26916,
            }
        )
        contexts, repeated, stats = generator.generate_batch(
            encoded_rows=row,
            metadata=metadata,
            rng=np.random.default_rng(4),
        )
        self.assertEqual(stats.generated_contexts, 1)
        self.assertTrue(np.array_equal(repeated, row))
        context = contexts[0]
        self.assertIsNotNone(context.trajectory_query)
        self.assertEqual(context.trajectory_query.query_type, "spatio_temporal")
        temporal = context.trajectory_query.temporal_predicates[0]
        spatial = context.trajectory_query.spatial_predicates[0]
        self.assertEqual(context.tokens[1], PredicateToken(PredicateOp.LESS_THAN, value=temporal.upper))
        self.assertEqual(context.tokens[2], PredicateToken(PredicateOp.GREATER_EQUAL, value=temporal.lower))
        self.assertEqual(context.tokens[3], PredicateToken.range(spatial.min_x, spatial.max_x))
        self.assertEqual(context.tokens[5], PredicateToken.range(spatial.min_x, spatial.max_x))
        self.assertTrue(
            context_satisfies_row_with_trajectory_semantics(context, row[0], metadata)
        )

    def test_semantic_columns_are_not_double_filtered(self) -> None:
        metadata = _pol_segment_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_compact_trajectory_index(
                tmpdir,
                metadata=metadata,
                trip_ids=(20,),
                segment_idx=(0,),
                t_s=(10.0,),
                t_e=(20.0,),
                s_x=(-1.0,),
                s_y=(1.0,),
                e_x=(3.0,),
                e_y=(1.0,),
                segment_varying_columns=(
                    "segments:segment_idx",
                    "segments:t_s",
                    "segments:t_e",
                    "segments:s_x",
                    "segments:s_y",
                    "segments:e_x",
                    "segments:e_y",
                ),
            )
            provider = CompactTrajectorySegmentIndex.from_directory(tmpdir)
            context = GeneratedTrainingContext(
                tokens=(
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.wildcard(),
                    PredicateToken.range(0.0, 2.0),
                    PredicateToken.range(0.0, 2.0),
                    PredicateToken.range(0.0, 2.0),
                    PredicateToken.range(0.0, 2.0),
                    PredicateToken.equal(1),
                    PredicateToken.equal(1),
                ),
                included_tables=frozenset({"trips", "segments"}),
                inverse_fanout_columns=frozenset(),
                ordinary_predicates={},
                trajectory_query=TrajectoryQuerySemantics(
                    scalar_predicates=(
                        ("segments:s_x", PredicateToken.range(0.0, 2.0)),
                        ("segments:e_x", PredicateToken.range(0.0, 2.0)),
                        ("segments:s_y", PredicateToken.range(0.0, 2.0)),
                        ("segments:e_y", PredicateToken.range(0.0, 2.0)),
                    ),
                    spatial_predicates=(SegmentSpatialPredicate(0.0, 0.0, 2.0, 2.0),),
                ),
            )
            result = provider.evaluate_batch(anchor_trajectory_ids=(20,), contexts=(context,))
            self.assertTrue(bool(result.eligible_mask[0]))
            self.assertEqual(int(result.multiplicities[0]), 1)

    def test_pol_query_adapter_temporal_semantic_parity(self) -> None:
        metadata = _pol_segment_metadata()
        rows = _encode_pol_segments(
            metadata,
            (
                (0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1, 1),
                (1, 10.0, 20.0, 3.0, 3.0, 4.0, 4.0, 1, 1),
                (2, 20.0, 30.0, 0.0, 3.0, 2.0, 3.0, 1, 1),
            ),
        )
        record = {
            "query_id": "temporal",
            "tables": ["trips", "segments"],
            "predicates": [
                {
                    "table": "segments",
                    "attribute": "segment_time",
                    "dimension": "temporal",
                    "type": "temporal_interval",
                    "mode": "temporal_overlap",
                    "lower": 10.0,
                    "upper": 20.0,
                }
            ],
        }
        context = pol_workload_record_to_context(record, metadata)
        self.assertEqual(context.tokens[1], PredicateToken(PredicateOp.LESS_THAN, value=20.0))
        self.assertEqual(context.tokens[2], PredicateToken(PredicateOp.GREATER_EQUAL, value=10.0))
        direct_mask = np.asarray(
            [
                context_satisfies_row_with_trajectory_semantics(context, row, metadata)
                for row in rows
            ],
            dtype=bool,
        )
        self.assertTrue(np.array_equal(direct_mask, [True, True, False]))
        oracle = ExactOracle(metadata, rows).exact_distinct_trajectory_count(
            context,
            trajectory_ids=(10, 10, 20),
            segment_ids=((10, 0), (10, 1), (20, 2)),
        )
        self.assertEqual(oracle.matching_segments_true, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            write_compact_trajectory_index(
                tmpdir,
                metadata=metadata,
                trip_ids=(10, 10, 20),
                segment_idx=(0, 1, 2),
                t_s=(0.0, 10.0, 20.0),
                t_e=(10.0, 20.0, 30.0),
                s_x=(0.0, 3.0, 0.0),
                s_y=(0.0, 3.0, 3.0),
                e_x=(1.0, 4.0, 2.0),
                e_y=(1.0, 4.0, 3.0),
                segment_varying_columns=(
                    "segments:segment_idx",
                    "segments:t_s",
                    "segments:t_e",
                    "segments:s_x",
                    "segments:s_y",
                    "segments:e_x",
                    "segments:e_y",
                ),
            )
            provider = CompactTrajectorySegmentIndex.from_directory(tmpdir)
            result = provider.evaluate_batch(anchor_trajectory_ids=(10,), contexts=(context,))
        self.assertTrue(bool(result.eligible_mask[0]))
        self.assertEqual(int(result.multiplicities[0]), 2)

    def test_pol_query_adapter_spatial_returns_unsupported_distinct_status(self) -> None:
        metadata = _pol_segment_metadata()
        rows = _encode_pol_segments(
            metadata,
            ((0, 0.0, 10.0, -1.0, 1.0, 3.0, 1.0, 1, 1),),
        )
        record = {
            "query_id": "spatial",
            "tables": ["trips", "segments"],
            "predicates": [
                {
                    "table": "segments",
                    "attribute": "segment_geom",
                    "dimension": "spatial",
                    "type": "geometry",
                    "mode": "spatial_intersects",
                    "min_x": 0.0,
                    "min_y": 0.0,
                    "max_x": 2.0,
                    "max_y": 2.0,
                    "srid": 26916,
                }
            ],
        }
        context = pol_workload_record_to_context(record, metadata)
        self.assertTrue(
            context_satisfies_row_with_trajectory_semantics(context, rows[0], metadata)
        )
        exact = ExactOracle(metadata, rows).exact_distinct_trajectory_count(
            context,
            trajectory_ids=(10,),
            segment_ids=((10, 0),),
        )
        self.assertEqual(exact.matching_segments_true, 1)
        result = evaluate_pol_distinct_record(
            record,
            metadata=metadata,
            estimator=object(),  # not used for fail-closed spatial status
            oracle=ExactOracle(metadata, rows),
            trajectory_ids=(10,),
            segment_ids=((10, 0),),
        )
        self.assertEqual(
            result.distinct_estimate_status,
            "unsupported_base_segment_spatial_measure",
        )
        self.assertIsNone(result.matching_segments_true)
        self.assertIsNone(result.distinct_trajectory_qerror)

    def test_pol_temporal_database_truth_metrics(self) -> None:
        metadata = _pol_segment_metadata()
        record = {
            "query_id": "q1",
            "tables": ["trips", "segments"],
            "predicates": [
                {
                    "table": "segments",
                    "attribute": "segment_time",
                    "dimension": "temporal",
                    "type": "temporal_interval",
                    "mode": "temporal_overlap",
                    "lower": 10.0,
                    "upper": 20.0,
                }
            ],
            "join_cardinality": 100,
            "entity_cardinality": 40,
            "entity_sql": "SELECT COUNT(DISTINCT t.trip_id) FROM ...",
        }
        result = evaluate_pol_distinct_record(
            record,
            metadata=metadata,
            estimator=_FakeDistinctEstimator(matching=80.0, dedup=0.5),
        )
        self.assertEqual(result.matching_segments_true, 100)
        self.assertEqual(result.distinct_trajectories_true, 40)
        self.assertEqual(result.database_matching_segments_true, 100)
        self.assertAlmostEqual(result.a_true, 0.4)
        self.assertAlmostEqual(result.matching_segment_qerror, 1.25)
        self.assertAlmostEqual(result.distinct_trajectory_qerror, 1.0)
        self.assertAlmostEqual(result.a_abs_error, 0.1)

    def test_zero_matching_database_truth_has_no_dedup_ratio(self) -> None:
        metadata = _pol_segment_metadata()
        record = {
            "query_id": "q_empty",
            "tables": ["trips", "segments"],
            "predicates": [
                {
                    "table": "segments",
                    "attribute": "segment_time",
                    "dimension": "temporal",
                    "type": "temporal_interval",
                    "mode": "temporal_overlap",
                    "lower": 10.0,
                    "upper": 20.0,
                }
            ],
            "join_cardinality": 0,
            "entity_cardinality": 0,
            "entity_sql": "SELECT COUNT(DISTINCT t.trip_id) FROM ...",
        }
        result = evaluate_pol_distinct_record(
            record,
            metadata=metadata,
            estimator=_FakeDistinctEstimator(matching=0.0, dedup=0.5),
        )
        self.assertEqual(result.matching_segments_true, 0)
        self.assertEqual(result.distinct_trajectories_true, 0)
        self.assertIsNone(result.a_true)
        self.assertIsNone(result.database_a_true)
        self.assertIsNone(result.a_abs_error)
        self.assertEqual(result.database_truth_status, "database_truth_available")

    def test_database_truth_precedes_fixture_truth(self) -> None:
        metadata = _pol_segment_metadata()
        rows = _encode_pol_segments(
            metadata,
            (
                (0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1, 1),
                (1, 10.0, 20.0, 3.0, 3.0, 4.0, 4.0, 1, 1),
                (2, 20.0, 30.0, 0.0, 3.0, 2.0, 3.0, 1, 1),
            ),
        )
        record = {
            "query_id": "q1",
            "tables": ["trips", "segments"],
            "predicates": [
                {
                    "table": "segments",
                    "attribute": "segment_time",
                    "dimension": "temporal",
                    "type": "temporal_interval",
                    "mode": "temporal_overlap",
                    "lower": 10.0,
                    "upper": 20.0,
                }
            ],
            "join_cardinality": 100,
            "entity_cardinality": 40,
            "entity_sql": "SELECT COUNT(DISTINCT t.trip_id) FROM ...",
        }
        result = evaluate_pol_distinct_record(
            record,
            metadata=metadata,
            estimator=_FakeDistinctEstimator(matching=80.0, dedup=0.5),
            oracle=ExactOracle(metadata, rows),
            trajectory_ids=(10, 10, 20),
            segment_ids=((10, 0), (10, 1), (20, 2)),
        )
        self.assertEqual(result.matching_segments_true, 100)
        self.assertEqual(result.distinct_trajectories_true, 40)
        self.assertEqual(result.fixture_matching_segments, 2)
        self.assertEqual(result.fixture_distinct_trajectories, 1)
        self.assertAlmostEqual(result.matching_segment_qerror, 1.25)
        self.assertAlmostEqual(result.distinct_trajectory_qerror, 1.0)

    def test_missing_database_truth_produces_no_production_qerror(self) -> None:
        metadata = _pol_segment_metadata()
        rows = _encode_pol_segments(
            metadata,
            (
                (0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1, 1),
                (1, 10.0, 20.0, 3.0, 3.0, 4.0, 4.0, 1, 1),
            ),
        )
        record = {
            "query_id": "q_missing_truth",
            "tables": ["trips", "segments"],
            "predicates": [
                {
                    "table": "segments",
                    "attribute": "segment_time",
                    "dimension": "temporal",
                    "type": "temporal_interval",
                    "mode": "temporal_overlap",
                    "lower": 10.0,
                    "upper": 20.0,
                }
            ],
            "join_cardinality": None,
            "entity_cardinality": None,
        }
        result = evaluate_pol_distinct_record(
            record,
            metadata=metadata,
            estimator=_FakeDistinctEstimator(matching=80.0, dedup=0.5),
            oracle=ExactOracle(metadata, rows),
            trajectory_ids=(10, 10),
            segment_ids=((10, 0), (10, 1)),
        )
        self.assertIsNone(result.matching_segments_true)
        self.assertIsNone(result.distinct_trajectories_true)
        self.assertIsNone(result.matching_segment_qerror)
        self.assertIsNone(result.distinct_trajectory_qerror)
        self.assertEqual(result.fixture_matching_segments, 2)
        self.assertEqual(result.fixture_distinct_trajectories, 1)

    def test_trip_geom_spatial_query_returns_unsupported_not_missing_column(self) -> None:
        metadata = _pol_segment_metadata()
        record = {
            "query_id": "trip_spatial",
            "tables": ["trips", "segments"],
            "predicates": [
                {
                    "table": "trips",
                    "attribute": "trip_geom",
                    "dimension": "spatial",
                    "type": "geometry",
                    "mode": "spatial_intersects",
                    "min_x": 0.0,
                    "min_y": 0.0,
                    "max_x": 2.0,
                    "max_y": 2.0,
                    "srid": 26916,
                }
            ],
            "join_cardinality": 100,
            "entity_cardinality": 40,
            "entity_sql": "SELECT COUNT(DISTINCT t.trip_id) FROM ...",
        }
        context = pol_workload_record_to_context(record, metadata)
        self.assertEqual(context.ordinary_predicates, {})
        self.assertIsNotNone(context.trajectory_query)
        spatial = context.trajectory_query.spatial_predicates[0]  # type: ignore[union-attr]
        self.assertIsInstance(spatial, PhysicalSpatialPredicate)
        self.assertEqual(spatial.geometry_column, "trips:trip_geom")
        result = evaluate_pol_distinct_record(
            record,
            metadata=metadata,
            estimator=_FakeDistinctEstimator(),
        )
        self.assertEqual(
            result.distinct_estimate_status,
            "unsupported_base_segment_spatial_measure",
        )
        self.assertEqual(result.matching_segments_true, 100)

    def test_checkpoint_runtime_trajectory_config_mismatch_fails(self) -> None:
        runtime = TrajectoryDistinctRuntimeConfig(
            segment_varying_columns=("segments:t_s", "segments:t_e"),
            srid=26916,
        )
        payload = {
            "trajectory_distinct": {
                "enabled": True,
                "segment_varying_columns": ["segments:t_s", "segments:t_e"],
                "srid": 26916,
            },
            "trajectory_target_semantics_version": TRAJECTORY_TARGET_SEMANTICS_VERSION,
        }
        assert_checkpoint_trajectory_config_compatible(payload, runtime)

        stale_config = {
            **payload,
            "trajectory_distinct": {
                **payload["trajectory_distinct"],
                "segment_key": "wrong_segment_key",
            },
        }
        with self.assertRaisesRegex(ValueError, "segment_key"):
            assert_checkpoint_trajectory_config_compatible(stale_config, runtime)

        stale_semantics = {
            **payload,
            "trajectory_target_semantics_version": "single_anchor_query_only_v1",
        }
        with self.assertRaisesRegex(ValueError, "semantics version"):
            assert_checkpoint_trajectory_config_compatible(stale_semantics, runtime)

    def test_runtime_config_mismatch_rejects_stale_index(self) -> None:
        metadata = _pol_segment_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_compact_trajectory_index(
                tmpdir,
                metadata=metadata,
                trip_ids=(10,),
                segment_idx=(0,),
                t_s=(0.0,),
                t_e=(10.0,),
                s_x=(0.0,),
                s_y=(0.0,),
                e_x=(1.0,),
                e_y=(1.0,),
                segment_varying_columns=("segments:segment_idx",),
            )
            provider = CompactTrajectorySegmentIndex.from_directory(tmpdir)
            with self.assertRaises(ValueError):
                provider.validate_runtime_compatibility(
                    TrajectoryDistinctRuntimeConfig(
                        segment_varying_columns=("segments:t_s",),
                    ),
                    metadata,
                )

    def test_fixture_startup_requires_hashable_segment_provenance(self) -> None:
        metadata = _pol_segment_metadata()
        rows = _encode_pol_segments(
            metadata,
            ((0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1, 1),),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "dataset_name": "pol_fixture",
                "dataset_type": "pol_trajectory_full_join",
                "join_cardinality": 1,
                "metadata": metadata.to_json_dict(),
                "domains_complete": True,
                "metadata_source": "complete_base_tables_and_join_metadata",
                "sample_rows": 1,
                "format_version": 2,
            }
            Path(tmpdir, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.save(Path(tmpdir, "sample_rows.npy"), rows)
            np.save(Path(tmpdir, "sample_trajectory_ids.npy"), np.asarray([10], dtype=np.int64))
            np.save(Path(tmpdir, "sample_segment_ids.npy"), np.asarray([[10, 0]], dtype=np.int64))
            write_compact_trajectory_index(
                Path(tmpdir, "trajectory_segment_index"),
                metadata=metadata,
                trip_ids=(10,),
                segment_idx=(0,),
                t_s=(0.0,),
                t_e=(10.0,),
                s_x=(0.0,),
                s_y=(0.0,),
                e_x=(1.0,),
                e_y=(1.0,),
                segment_varying_columns=(
                    "segments:segment_idx",
                    "segments:t_s",
                    "segments:t_e",
                    "segments:s_x",
                    "segments:s_y",
                    "segments:e_x",
                    "segments:e_y",
                ),
            )
            source = NeuroCardFullJoinSampleSource(tmpdir)
            batch = source.batches(1, seed=0)
            self.assertIsInstance(batch.segment_ids[0], tuple)  # type: ignore[index]
            self.assertEqual(hash(batch.segment_ids[0]), hash((10, 0)))  # type: ignore[index]
            source.validate_trajectory_distinct(
                runtime_config=TrajectoryDistinctRuntimeConfig(
                    segment_varying_columns=(
                        "segments:segment_idx",
                        "segments:t_s",
                        "segments:t_e",
                        "segments:s_x",
                        "segments:s_y",
                        "segments:e_x",
                        "segments:e_y",
                    ),
                )
            )
            Path(tmpdir, "sample_segment_ids.npy").unlink()
            missing = NeuroCardFullJoinSampleSource(tmpdir)
            with self.assertRaises(ValueError):
                missing.validate_trajectory_distinct(
                    runtime_config=TrajectoryDistinctRuntimeConfig(
                        segment_varying_columns=(
                            "segments:segment_idx",
                            "segments:t_s",
                            "segments:t_e",
                            "segments:s_x",
                            "segments:s_y",
                            "segments:e_x",
                            "segments:e_y",
                        ),
                    )
                )

    def test_prepare_pol_staging_data_builds_fixture_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging = root / "staging"
            prepared = root / "prepared"
            staging.mkdir()
            (staging / "agents.tsv").write_text(
                "\n".join(
                    [
                        "1\t30.0\tGraduate\tA\t0.5\t2",
                        "2\t40.0\tHighSchoolOrCollege\tB\t0.7\t3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (staging / "trips.tsv").write_text(
                "\n".join(
                    [
                        "10\t1\t2020-01-01T00:00:00.000\t2020-01-01T00:05:00.000\t2",
                        "20\t2\t2020-01-02T00:00:00.000\t2020-01-02T00:05:00.000\t1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (staging / "segments.tsv").write_text(
                "\n".join(
                    [
                        "10\t0\t0.0\t0.0\t1.0\t1.0\t2020-01-01T00:00:00.000\t2020-01-01T00:01:00.000",
                        "10\t1\t1.0\t1.0\t2.0\t2.0\t2020-01-01T00:01:00.000\t2020-01-01T00:02:00.000",
                        "20\t0\t2.0\t2.0\t3.0\t3.0\t2020-01-02T00:00:00.000\t2020-01-02T00:01:00.000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                f"""
dataset:
  type: pol_trajectory_full_join
  name: pol_test
  prepared_directory: {prepared}
  sampling_mode: fixture
  trajectory_index_path: {prepared}/trajectory_segment_index

factorization:
  enabled: false

trajectory_distinct:
  enabled: true
  entity_table: trips
  segment_table: segments
  trajectory_key: trip_id
  segment_key: trip_id,segment_idx
  predicate_scope: segment_query
  srid: 26916
  trajectory_static_columns:
    - agents:age
    - trips:trip_geom
  segment_varying_columns:
    - segments:segment_idx
    - segments:t_s
    - segments:t_e
    - segments:s_x
    - segments:s_y
    - segments:e_x
    - segments:e_y
""",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "model.scripts.prepare_pol_staging_data",
                    "--config",
                    str(config_path),
                    "--staging-dir",
                    str(staging),
                    "--sample-rows",
                    "2",
                    "--seed",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("prepared_directory=", completed.stdout)
            source = NeuroCardFullJoinSampleSource(prepared)
            self.assertEqual(len(source.metadata.columns), 21)
            self.assertIn("trips:trip_geom", [column.name for column in source.metadata.columns])
            self.assertEqual(np.load(prepared / "sample_rows.npy", mmap_mode="r").shape, (2, 21))
            self.assertEqual(np.load(prepared / "sample_trajectory_ids.npy").shape, (2,))
            self.assertEqual(np.load(prepared / "sample_segment_ids.npy").shape, (2, 2))
            source.validate_trajectory_distinct(
                runtime_config=TrajectoryDistinctRuntimeConfig(
                    segment_varying_columns=(
                        "segments:segment_idx",
                        "segments:t_s",
                        "segments:t_e",
                        "segments:s_x",
                        "segments:s_y",
                        "segments:e_x",
                        "segments:e_y",
                    ),
                    trajectory_static_columns=("agents:age", "trips:trip_geom"),
                    srid=26916,
                )
            )

    def test_exact_oracle_uses_trajectory_semantics(self) -> None:
        metadata = _pol_segment_metadata()
        rows = _encode_pol_segments(
            metadata,
            (
                (0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1, 1),
                (1, 10.0, 20.0, 3.0, 3.0, 4.0, 4.0, 1, 1),
                (2, 20.0, 30.0, 0.0, 3.0, 2.0, 3.0, 1, 1),
            ),
        )
        context = GeneratedTrainingContext(
            tokens=tuple(PredicateToken.wildcard() for _ in metadata.columns),
            included_tables=frozenset({"trips", "segments"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={},
            trajectory_query=TrajectoryQuerySemantics(
                temporal_predicates=(
                    SegmentTemporalPredicate("segments:t_s", "segments:t_e", lower=10.0, upper=25.0),
                ),
                spatial_predicates=(
                    SegmentSpatialPredicate(0.0, 2.5, 2.0, 3.5),
                ),
            ),
        )
        result = ExactOracle(metadata, rows).exact_distinct_trajectory_count(
            context,
            trajectory_ids=(10, 10, 20),
            segment_ids=((10, 0), (10, 1), (20, 2)),
        )
        self.assertEqual(result.matching_segments_true, 1)
        self.assertEqual(result.distinct_trajectories_true, 1)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TrajectoryDistinctTorchTest(unittest.TestCase):

    def test_weighted_mse_uses_independent_terminal_denominator(self) -> None:
        metadata = _trajectory_metadata()
        rows, trajectory_ids = _trajectory_rows(metadata)
        provider = TrajectorySegmentIndex.from_rows(
            metadata=metadata,
            trajectory_ids=trajectory_ids,
            encoded_rows=rows,
            predicate_columns=("segment.x",),
            trajectory_key="trip_id",
        )
        source = _TrajectorySource(metadata, rows, trajectory_ids, provider)
        batch = FullJoinBatch(
            encoded_values=rows[[0, 2]],
            column_metadata=metadata.columns,
            trajectory_ids=("A", "B"),
            importance_weights=np.asarray([2.0, 5.0]),
        )
        base_context = _query_context()
        tokens = list(base_context.tokens)
        tokens[-1] = PredicateToken.inv_fanout()
        context = GeneratedTrainingContext(
            tokens=tuple(tokens),
            included_tables=base_context.included_tables,
            inverse_fanout_columns=frozenset({"F_trips_to_segments"}),
            ordinary_predicates=base_context.ordinary_predicates,
        )
        predictions = torch.tensor([[0.25], [0.25]], dtype=torch.float32, requires_grad=True)
        loss, stats = trajectory_dedup_loss_for_batch(
            predictions=predictions,
            batch=batch,
            contexts=[context, context],
            target_rows=batch.encoded_values,
            token_rows=[list(context.tokens), list(context.tokens)],
            generation_stats=type("Stats", (), {"source_row_indices": (0, 1)})(),
            metadata=metadata,
            config={"fanout": {"compute_weights_in_log_space": True}},
            sample_source=source,
            device="cpu",
        )
        expected = ((0.25 - 1.0) ** 2 + (0.25 - 0.5) ** 2 * 0.5) / 1.5
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.assertAlmostEqual(stats["weighted_ess"], effective_sample_size(np.asarray([1.0, 0.5])))
        loss.backward()
        self.assertIsNotNone(predictions.grad)

    def test_importance_batch_scaling_does_not_change_global_metric(self) -> None:
        batches = [
            (np.asarray([1.0, 2.0]), np.asarray([0.25, 1.0])),
            (np.asarray([1000.0, 2000.0]), np.asarray([4.0, 9.0])),
        ]
        running = _RunningTrajectoryDistinctStats()
        all_weights = []
        all_errors = []
        for weights, squared_errors in batches:
            all_weights.extend(weights.tolist())
            all_errors.extend(squared_errors.tolist())
            stable = weights / float(np.max(weights))
            running.update(
                {
                    "enabled": True,
                    "trajectory_targets_attempted": len(weights),
                    "trajectory_targets_generated": len(weights),
                    "trajectory_targets_skipped": 0,
                    "multiplicity_sum": float(len(weights)),
                    "m_min": 1,
                    "m_max": 1,
                    "weighted_mse": float(np.dot(stable, squared_errors) / np.sum(stable)),
                    "unweighted_mse": float(np.mean(squared_errors)),
                    "weighted_squared_error_sum": float(np.dot(stable, squared_errors)),
                    "unweighted_squared_error_sum": float(np.sum(squared_errors)),
                    "traj_loss_weight_sum": float(np.sum(stable)),
                    "traj_loss_weight_sq_sum": float(np.dot(stable, stable)),
                    "traj_log_weight_sum": float(np.logaddexp.reduce(np.log(weights))),
                    "traj_log_weight_sq_sum": float(np.logaddexp.reduce(2.0 * np.log(weights))),
                    "traj_log_weighted_squared_error_sum": float(
                        np.logaddexp.reduce(np.log(weights) + np.log(squared_errors))
                    ),
                }
            )
        summary = running.to_json_dict()
        all_weights_np = np.asarray(all_weights)
        all_errors_np = np.asarray(all_errors)
        expected_mse = float(np.dot(all_weights_np, all_errors_np) / np.sum(all_weights_np))
        expected_ess = effective_sample_size(all_weights_np)
        self.assertAlmostEqual(summary["global_weighted_mse"], expected_mse, places=10)
        self.assertAlmostEqual(summary["global_weighted_ess"], expected_ess, places=10)

    def test_global_weighted_mse_exact_zero(self) -> None:
        running = _RunningTrajectoryDistinctStats()
        for weights in (np.asarray([1.0, 2.0]), np.asarray([1000.0, 2000.0])):
            running.update(
                {
                    "enabled": True,
                    "trajectory_targets_attempted": len(weights),
                    "trajectory_targets_generated": len(weights),
                    "trajectory_targets_skipped": 0,
                    "multiplicity_sum": float(len(weights)),
                    "m_min": 1,
                    "m_max": 1,
                    "weighted_mse": 0.0,
                    "unweighted_mse": 0.0,
                    "weighted_squared_error_sum": 0.0,
                    "unweighted_squared_error_sum": 0.0,
                    "traj_loss_weight_sum": float(np.sum(weights / np.max(weights))),
                    "traj_loss_weight_sq_sum": float(np.dot(weights / np.max(weights), weights / np.max(weights))),
                    "traj_log_weight_sum": float(np.logaddexp.reduce(np.log(weights))),
                    "traj_log_weight_sq_sum": float(np.logaddexp.reduce(2.0 * np.log(weights))),
                    "traj_log_weighted_squared_error_sum": float("-inf"),
                }
            )
        summary = running.to_json_dict()
        self.assertEqual(summary["global_weighted_mse"], 0.0)
        self.assertEqual(summary["global_unweighted_mse"], 0.0)

    def test_spatial_trajectory_target_is_skipped_when_base_measure_unsupported(self) -> None:
        metadata = _pol_segment_metadata()
        rows = _encode_pol_segments(
            metadata,
            ((0, 0.0, 10.0, -1.0, 1.0, 3.0, 1.0, 1, 1),),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_compact_trajectory_index(
                tmpdir,
                metadata=metadata,
                trip_ids=(10,),
                segment_idx=(0,),
                t_s=(0.0,),
                t_e=(10.0,),
                s_x=(-1.0,),
                s_y=(1.0,),
                e_x=(3.0,),
                e_y=(1.0,),
                segment_varying_columns=(
                    "segments:segment_idx",
                    "segments:t_s",
                    "segments:t_e",
                    "segments:s_x",
                    "segments:s_y",
                    "segments:e_x",
                    "segments:e_y",
                ),
            )
            provider = CompactTrajectorySegmentIndex.from_directory(tmpdir)
            source = _TrajectorySource(metadata, rows, ("10",), provider)  # type: ignore[arg-type]
            context = GeneratedTrainingContext(
                tokens=tuple(PredicateToken.wildcard() for _ in metadata.columns),
                included_tables=frozenset({"trips", "segments"}),
                inverse_fanout_columns=frozenset(),
                ordinary_predicates={},
                trajectory_query=TrajectoryQuerySemantics(
                    spatial_predicates=(SegmentSpatialPredicate(0.0, 0.0, 2.0, 2.0),),
                ),
            )
            batch = FullJoinBatch(
                encoded_values=rows,
                column_metadata=metadata.columns,
                trajectory_ids=(10,),
            )
            predictions = torch.tensor([[0.25]], dtype=torch.float32, requires_grad=True)
            loss, stats = trajectory_dedup_loss_for_batch(
                predictions=predictions,
                batch=batch,
                contexts=[context],
                target_rows=rows,
                token_rows=[list(context.tokens)],
                generation_stats=type("Stats", (), {"source_row_indices": (0,)})(),
                metadata=metadata,
                config={"fanout": {"compute_weights_in_log_space": True}},
                sample_source=source,
                device="cpu",
            )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(stats["trajectory_targets_generated"], 0)
        self.assertEqual(stats["trajectory_targets_skipped_base_spatial_measure"], 1)

    def test_run_validation_two_batches_returns_finite_traj_metrics(self) -> None:
        metadata = _trajectory_metadata()
        rows, trajectory_ids = _trajectory_rows(metadata)
        provider = TrajectorySegmentIndex.from_rows(
            metadata=metadata,
            trajectory_ids=trajectory_ids,
            encoded_rows=rows,
            trajectory_key="trip_id",
            trajectory_static_columns=("trip.kind",),
            segment_varying_columns=("segment.x", "segment.y"),
            segment_table="segments",
        )
        source = _TrajectorySource(metadata, rows, trajectory_ids, provider)
        vocab = PredicateVocabularies.from_metadata(metadata)
        model = PredicateResMADE(
            PredicateResMADEConfig(
                predicate_input_bins=vocab.input_bins,
                data_output_bins=metadata.data_output_bins,
                column_kinds=tuple(column.kind.value for column in metadata.columns),
                hidden_sizes=(16, 16),
                direct_io_connections=True,
                embedding_size=4,
                trajectory_distinct_config=TrajectoryDistinctConfig(enabled=True),
            )
        )
        config = {
            "training": {"seed": 0, "head_loss_reduction": "sum"},
            "fanout": {"compute_weights_in_log_space": True},
            "anpm": {"mask_invalid_combinations": True},
            "predicate_generation": {
                "enabled": True,
                "strategy": "row_satisfied",
                "table_subset_sampling": "full",
                "wildcard_probability": 1.0,
                "equality_probability": 0.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "trajectory_query_semantics": "none",
            },
            "trajectory_distinct": {
                "enabled": True,
                "segment_table": "segments",
                "trajectory_key": "trip_id",
                "segment_key": "trip_id,segment_idx",
                "predicate_scope": "segment_query",
            },
        }
        metrics = _run_validation(
            model,
            source,
            metadata,
            vocab,
            config,
            "cpu",
            batch_size=2,
            batches=2,
            seed=0,
        )
        self.assertGreater(metrics["validation_traj_target_count"], 0)
        self.assertTrue(np.isfinite(metrics["validation_traj_weighted_mse"]))
        self.assertTrue(np.isfinite(metrics["validation_traj_unweighted_mse"]))
        self.assertEqual(metrics["validation_source_mode"], "same_fixture_resampled")

    def test_held_out_validation_source_uses_prepared_directory(self) -> None:
        metadata = _pol_segment_metadata()
        train_rows = _encode_pol_segments(
            metadata,
            ((0, 0.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1, 1),),
        )
        validation_rows = _encode_pol_segments(
            metadata,
            ((1, 10.0, 20.0, 3.0, 3.0, 4.0, 4.0, 1, 1),),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            _write_prepared_pol_fixture(
                train_dir,
                metadata=metadata,
                rows=train_rows,
                trajectory_ids=(10,),
                segment_ids=((10, 0),),
            )
            _write_prepared_pol_fixture(
                validation_dir,
                metadata=metadata,
                rows=validation_rows,
                trajectory_ids=(20,),
                segment_ids=((20, 1),),
            )
            config = {
                "dataset": {
                    "type": "pol_trajectory_full_join",
                    "prepared_directory": str(train_dir),
                    "sampling_mode": "fixture",
                    "trajectory_index_path": str(train_dir / "trajectory_segment_index"),
                },
                "trajectory_distinct": {
                    "enabled": True,
                    "segment_table": "segments",
                    "trajectory_key": "trip_id",
                    "segment_key": "trip_id,segment_idx",
                    "predicate_scope": "segment_query",
                    "segment_varying_columns": [
                        "segments:segment_idx",
                        "segments:t_s",
                        "segments:t_e",
                        "segments:s_x",
                        "segments:s_y",
                        "segments:e_x",
                        "segments:e_y",
                    ],
                },
            }
            validation_source = _validation_sample_source_from_config(
                config,
                {
                    "prepared_directory": str(validation_dir),
                    "trajectory_index_path": str(
                        validation_dir / "trajectory_segment_index"
                    ),
                },
            )
            batch = validation_source.batches(1, seed=0)
            self.assertTrue(np.array_equal(batch.encoded_values, validation_rows))
            self.assertEqual(batch.trajectory_ids, (20,))
            self.assertEqual(batch.segment_ids, ((20, 1),))

    def test_validation_selection_metric_fails_when_no_eligible_targets(self) -> None:
        metadata = _trajectory_metadata()
        rows, trajectory_ids = _trajectory_rows(metadata)
        no_segment_rows = rows.copy()
        no_segment_rows[:, 4] = metadata.columns[4].encode_value(0)
        provider = TrajectorySegmentIndex.from_rows(
            metadata=metadata,
            trajectory_ids=trajectory_ids,
            encoded_rows=rows,
            trajectory_key="trip_id",
            segment_varying_columns=("segment.x", "segment.y"),
            segment_table="segments",
        )
        source = _TrajectorySource(metadata, no_segment_rows, trajectory_ids, provider)
        config = {
            "model": {
                "type": "predicate_resmade",
                "hidden_sizes": [16],
                "residual_connections": True,
                "direct_io_connections": True,
                "embedding_size": 4,
            },
            "predicate_generation": {
                "enabled": True,
                "strategy": "row_satisfied",
                "table_subset_sampling": "full",
                "wildcard_probability": 1.0,
                "equality_probability": 0.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "trajectory_query_semantics": "none",
            },
            "fanout": {"compute_weights_in_log_space": True},
            "training": {
                "seed": 0,
                "device": "cpu",
                "batch_size": 2,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 1,
                "steps_per_epoch": 1,
                "checkpoint_interval_steps": 0,
                "validation_interval_steps": 1,
                "head_loss_reduction": "sum",
            },
            "validation": {
                "enabled": True,
                "interval_steps": 1,
                "fresh_sampler_batches": 2,
                "batch_size": 2,
                "selection_metric": "validation_traj_weighted_mse",
                "minimize": True,
            },
            "inference": {"use_log_space_product": True},
            "trajectory_distinct": {
                "enabled": True,
                "head_name": "traj_dedup_factor",
                "loss": "mse",
                "loss_weight": 1.0,
                "output_activation": "sigmoid",
                "anchor_samples_per_query": 1,
                "entity_table": "trips",
                "segment_table": "segments",
                "trajectory_key": "trip_id",
                "segment_key": "trip_id,segment_idx",
                "predicate_scope": "segment_query",
                "segment_varying_columns": ["segment.x", "segment.y"],
            },
            "factorization": {"enabled": False},
            "logging": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config["logging"]["output_directory"] = tmpdir
            with self.assertRaisesRegex(
                ValueError,
                "validation selection metric validation_traj_weighted_mse is unavailable",
            ):
                train_resmade_sample_source(source, config)

    def test_one_forward_distinct_inference_and_checkpoint_guard(self) -> None:
        metadata = _trajectory_metadata()
        vocab = PredicateVocabularies.from_metadata(metadata)
        config = PredicateResMADEConfig(
            predicate_input_bins=vocab.input_bins,
            data_output_bins=metadata.data_output_bins,
            column_kinds=tuple(column.kind.value for column in metadata.columns),
            hidden_sizes=(16, 16),
            direct_io_connections=True,
            embedding_size=4,
            trajectory_distinct_config=TrajectoryDistinctConfig(enabled=True),
        )
        model = PredicateResMADE(config)
        wrapped = TorchDistributionModel(model, metadata, vocab, device="cpu")
        estimator = OnePassEstimator(wrapped, metadata)
        result = estimator.estimate_distinct_trajectories(
            list(_query_context().tokens),
            context=_query_context(),
        )
        self.assertEqual(result.model_forward_calls, 1)
        self.assertGreater(result.matching_segment_estimate, 0.0)
        self.assertGreater(result.traj_dedup_factor, 0.0)
        self.assertLess(result.traj_dedup_factor, 1.0)
        self.assertLessEqual(
            result.distinct_trajectory_estimate,
            result.matching_segment_estimate,
        )
        with self.assertRaises(TrajectoryDistinctNotApplicable):
            estimator.estimate_distinct_trajectories(
                list(_query_context().tokens),
                included_tables={"trips", "segments"},
            )

        disabled = PredicateResMADE(
            PredicateResMADEConfig(
                predicate_input_bins=vocab.input_bins,
                data_output_bins=metadata.data_output_bins,
                hidden_sizes=(16, 16),
                embedding_size=4,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/checkpoint.pt"
            save_resmade_checkpoint(
                path,
                disabled,
                None,
                epoch=0,
                step=0,
                metadata=metadata,
                predicate_vocabularies=vocab,
                config={"trajectory_distinct": {"enabled": False}},
            )
            loaded, _ = load_resmade_checkpoint(path)
        wrapped_disabled = TorchDistributionModel(loaded, metadata, vocab, device="cpu")
        with self.assertRaises(ValueError):
            OnePassEstimator(wrapped_disabled, metadata).estimate_distinct_trajectories(
                list(_query_context().tokens),
                context=_query_context(),
            )

    def test_spatial_distinct_inference_fails_closed(self) -> None:
        metadata = _pol_segment_metadata()
        vocab = PredicateVocabularies.from_metadata(metadata)
        model = PredicateResMADE(
            PredicateResMADEConfig(
                predicate_input_bins=vocab.input_bins,
                data_output_bins=metadata.data_output_bins,
                column_kinds=tuple(column.kind.value for column in metadata.columns),
                hidden_sizes=(16, 16),
                direct_io_connections=True,
                embedding_size=4,
                trajectory_distinct_config=TrajectoryDistinctConfig(enabled=True),
            )
        )
        wrapped = TorchDistributionModel(model, metadata, vocab, device="cpu")
        context = GeneratedTrainingContext(
            tokens=tuple(PredicateToken.wildcard() for _ in metadata.columns),
            included_tables=frozenset({"trips", "segments"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={},
            trajectory_query=TrajectoryQuerySemantics(
                spatial_predicates=(SegmentSpatialPredicate(0.0, 0.0, 2.0, 2.0),),
            ),
        )
        with self.assertRaises(TrajectoryDistinctNotApplicable) as caught:
            OnePassEstimator(wrapped, metadata).estimate_distinct_trajectories(
                list(context.tokens),
                context=context,
            )
        self.assertEqual(str(caught.exception), "unsupported_base_segment_spatial_measure")


if __name__ == "__main__":
    unittest.main()
