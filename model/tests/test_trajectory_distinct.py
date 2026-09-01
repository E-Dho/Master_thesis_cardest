from __future__ import annotations

import importlib.util
import json
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
    SegmentSpatialPredicate,
    SegmentTemporalPredicate,
    TrajectoryDistinctRuntimeConfig,
    TrajectorySegmentIndex,
    TrajectoryQuerySemantics,
    UnsupportedTrajectoryContext,
    context_satisfies_row_with_trajectory_semantics,
    segment_rectangle_intersects_mask,
    temporal_overlap_mask,
    trajectory_distinct_context_eligibility,
    write_compact_trajectory_index,
)
from model.src.evaluation.exact_evaluator import ExactOracle
from model.src.predicates.generation import GeneratedTrainingContext, PredicateTrainingContextGenerator
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.vocabulary import PredicateVocabularies
from model.src.training.losses import (
    effective_sample_size,
    stable_combine_importance_and_terminal_inverse_weights,
    terminal_inverse_fanout_weights,
)

if HAS_TORCH:
    from model.src.inference.estimator import OnePassEstimator
    from model.src.inference.torch_estimator import TorchDistributionModel
    from model.src.model.checkpoint import load_resmade_checkpoint, save_resmade_checkpoint
    from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
    from model.src.model.resmade import TrajectoryDistinctConfig
    from model.src.training.resmade_trainer import trajectory_dedup_loss_for_batch


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
        ColumnMetadata("segments:s_x", ColumnKind.DATA, (0.0, 1.0, 3.0), table="segments"),
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


if __name__ == "__main__":
    unittest.main()
