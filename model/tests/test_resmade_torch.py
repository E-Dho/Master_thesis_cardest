from __future__ import annotations

import importlib.util
import tempfile
import unittest

import numpy as np

from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
from model.src.predicates.generation import tokens_for_query_tables
from model.src.predicates.vocabulary import PredicateVocabularies
from model.src.training.losses import cumulative_inverse_fanout_weights, weighted_cross_entropy

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is not installed")

import torch

from model.src.config import load_simple_yaml
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.model.anpm import ANPMColumnDecoder, ANPMConfig
from model.src.model.checkpoint import load_resmade_checkpoint, save_resmade_checkpoint
from model.src.model.factorization import FactorizationConfig, apply_factorization_to_metadata
from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.predicates.operators import PredicateToken
from model.src.training.torch_losses import torch_weighted_per_head_cross_entropy
from model.src.inference.torch_estimator import TorchDistributionModel


class ResMADETorchTest(unittest.TestCase):
    def _model(self, residual: bool = True, direct_io: bool = True) -> tuple[PredicateResMADE, SyntheticFullJoinSampleSource, PredicateVocabularies]:
        source = SyntheticFullJoinSampleSource()
        vocabularies = PredicateVocabularies.from_metadata(source.metadata)
        model = PredicateResMADE(
            PredicateResMADEConfig(
                predicate_input_bins=vocabularies.input_bins,
                data_output_bins=source.metadata.data_output_bins,
                hidden_sizes=(32, 32),
                residual_connections=residual,
                direct_io_connections=direct_io,
                embedding_size=8,
            )
        )
        return model, source, vocabularies

    def test_output_width_slices_and_softmax(self) -> None:
        model, source, vocabularies = self._model()
        tokens = [[tokens_for_query_tables(source.metadata, {"A"}, {"F_A_to_B"})[i] for i in range(len(source.metadata.columns))]]
        token_ids = encode_tokens_tensor(tokens, vocabularies)
        logits = model(token_ids)
        self.assertEqual(logits.shape[1], sum(source.metadata.data_output_bins))
        distributions = model.predict_distributions(token_ids)
        self.assertEqual(len(distributions), len(source.metadata.columns))
        for distribution, width in zip(distributions, source.metadata.data_output_bins):
            self.assertEqual(distribution.shape, (1, width))
            self.assertTrue(torch.allclose(distribution.sum(dim=1), torch.ones(1)))

    def test_autoregressive_no_current_or_future_leakage(self) -> None:
        model, source, vocabularies = self._model(residual=True, direct_io=True)
        tokens_a = tokens_for_query_tables(source.metadata, {"A"}, {"F_A_to_B"})
        tokens_b = list(tokens_a)
        tokens_b[0] = tokens_for_query_tables(source.metadata, {"A"}, set())[0]
        ids_a = encode_tokens_tensor([tokens_a], vocabularies)
        ids_b = encode_tokens_tensor([tokens_b], vocabularies)
        logits_a = model(ids_a)
        logits_b = model(ids_b)
        start, stop = source.metadata.output_slices[0]
        self.assertTrue(torch.allclose(logits_a[:, start:stop], logits_b[:, start:stop]))

        tokens_c = list(tokens_a)
        tokens_c[-1] = tokens_for_query_tables(source.metadata, {"A"}, set())[-1]
        logits_c = model(encode_tokens_tensor([tokens_c], vocabularies))
        self.assertTrue(torch.allclose(logits_a[:, start:stop], logits_c[:, start:stop]))

    def test_previous_token_can_affect_later_logits(self) -> None:
        model, source, vocabularies = self._model()
        tokens_a = tokens_for_query_tables(source.metadata, {"A"}, {"F_A_to_B"})
        tokens_b = tokens_for_query_tables(source.metadata, {"A", "B"}, {"F_A_to_B"})
        logits_a = model(encode_tokens_tensor([tokens_a], vocabularies))
        logits_b = model(encode_tokens_tensor([tokens_b], vocabularies))
        start, stop = source.metadata.output_slices[-1]
        self.assertFalse(torch.allclose(logits_a[:, start:stop], logits_b[:, start:stop]))

    def test_torch_weighted_loss_matches_numpy_and_gradients(self) -> None:
        logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
        metadata = SyntheticFullJoinSampleSource().metadata
        tiny_metadata = type(metadata)(
            columns=metadata.columns[:1],
            full_join_cardinality=2,
        )
        targets = torch.tensor([[0], [1]])
        weights = torch.tensor([[1.0], [3.0]])
        loss = torch_weighted_per_head_cross_entropy(logits, targets, weights, tiny_metadata)
        probabilities = torch.softmax(logits, dim=1).detach().numpy()
        expected = weighted_cross_entropy(probabilities, np.array([0, 1]), np.array([1.0, 3.0]))
        self.assertAlmostEqual(float(loss.total_loss.detach()), expected, places=6)
        loss.total_loss.backward()
        self.assertTrue(torch.all(torch.isfinite(logits.grad)))

    def test_checkpoint_roundtrip_preserves_ordering(self) -> None:
        model, source, vocabularies = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/checkpoint.pt"
            save_resmade_checkpoint(
                path,
                model,
                optimizer,
                epoch=0,
                step=1,
                metadata=source.metadata,
                predicate_vocabularies=vocabularies,
                config=load_simple_yaml("model/configs/resmade_smoke.yaml"),
            )
            loaded, payload = load_resmade_checkpoint(path)
        self.assertEqual(loaded.output_slices, model.output_slices)
        self.assertEqual(payload["metadata"]["column_order"], "data_indicators_fanouts")

    def test_factorized_resmade_rejects_direct_io(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        with self.assertRaises(ValueError):
            PredicateResMADE(
                PredicateResMADEConfig(
                    predicate_input_bins=vocabularies.input_bins,
                    data_output_bins=metadata.data_output_bins,
                    hidden_sizes=(16, 16),
                    direct_io_connections=True,
                    output_head_specs=metadata.factorization_plan.output_head_specs,
                    factorization_plan=metadata.factorization_plan,
                    anpm_config=ANPMConfig(enabled=True),
                )
            )

    def test_factorized_heads_use_original_column_degrees(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        tokens_a = [PredicateToken.equal(5), PredicateToken.equal("b")]
        tokens_b = [PredicateToken.equal(6), PredicateToken.equal("b")]
        tokens_c = [PredicateToken.equal(5), PredicateToken.equal("a")]
        logits_a = model(encode_tokens_tensor([tokens_a], vocabularies))
        logits_b = model(encode_tokens_tensor([tokens_b], vocabularies))
        logits_c = model(encode_tokens_tensor([tokens_c], vocabularies))
        for head_index in metadata.factorization_plan.output_heads_for_column(0):
            start, stop = model.output_slices[head_index]
            self.assertTrue(torch.allclose(logits_a[:, start:stop], logits_b[:, start:stop]))
            self.assertTrue(torch.allclose(logits_a[:, start:stop], logits_c[:, start:stop]))

    def test_anpm_prefix_dependency_and_grouped_loss_gradients(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        tokens = [[PredicateToken.wildcard(), PredicateToken.wildcard()]]
        token_ids = encode_tokens_tensor(tokens, vocabularies)
        rows = torch.tensor([[0, 0], [19, 1], [7, 0], [13, 1]])
        weights = torch.ones_like(rows, dtype=torch.float32)
        logits = model(token_ids.expand(rows.shape[0], -1))
        breakdown = torch_weighted_per_head_cross_entropy(
            logits,
            rows,
            weights,
            metadata,
            anpm_decoders=model.anpm_decoders,
        )
        breakdown.total_loss.backward()
        anpm_grads = [
            parameter.grad
            for parameter in model.anpm_decoders.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(anpm_grads)
        self.assertTrue(all(torch.all(torch.isfinite(grad)) for grad in anpm_grads))

        decoder = ANPMColumnDecoder(
            factor_domains=(4, 8),
            embedding_size=8,
            hidden_size=8,
        )
        base = torch.zeros(2, 8)
        logits_0 = decoder.conditional_logits(1, base, torch.tensor([[0], [1]]))
        self.assertFalse(torch.allclose(logits_0[0], logits_0[1]))

    def test_factorized_adapter_normalizes_original_distribution(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        wrapped = TorchDistributionModel(model, metadata, vocabularies)
        tokens = [PredicateToken.equal(5), PredicateToken.wildcard()]
        distributions = wrapped.predict_distributions(tokens)
        self.assertTrue(np.allclose(np.sum(distributions[0]), 1.0, atol=1.0e-5))
        factors = wrapped.predict_column_factors(tokens)
        self.assertGreaterEqual(factors[0], 0.0)
        self.assertLessEqual(factors[0], 1.0)

    def test_factorized_checkpoint_roundtrip_preserves_predictions(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        tokens = [[PredicateToken.wildcard(), PredicateToken.wildcard()]]
        token_ids = encode_tokens_tensor(tokens, vocabularies)
        expected_logits = model(token_ids).detach()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/factorized.pt"
            save_resmade_checkpoint(
                path,
                model,
                optimizer,
                epoch=0,
                step=1,
                metadata=metadata,
                predicate_vocabularies=vocabularies,
                config=load_simple_yaml("model/configs/resmade_factorized_smoke.yaml"),
            )
            loaded, payload = load_resmade_checkpoint(
                path,
                expected_factorization_plan=metadata.factorization_plan,
            )
        self.assertTrue(torch.allclose(expected_logits, loaded(token_ids)))
        self.assertTrue(payload["metadata"]["factorization_plan"]["enabled"])

    @staticmethod
    def _factorized_metadata() -> ModelMetadata:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("x", ColumnKind.DATA, tuple(range(20))),
                ColumnMetadata("y", ColumnKind.DATA, ("a", "b")),
            ),
            full_join_cardinality=20,
        )
        return apply_factorization_to_metadata(
            metadata,
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=3,
                minimum_domain_size=2,
            ),
        )

    @staticmethod
    def _factorized_model(
        metadata: ModelMetadata, vocabularies: PredicateVocabularies
    ) -> PredicateResMADE:
        return PredicateResMADE(
            PredicateResMADEConfig(
                predicate_input_bins=vocabularies.input_bins,
                data_output_bins=metadata.data_output_bins,
                hidden_sizes=(32, 32),
                direct_io_connections=False,
                embedding_size=8,
                output_head_specs=metadata.factorization_plan.output_head_specs,
                factorization_plan=metadata.factorization_plan,
                anpm_config=ANPMConfig(
                    enabled=True,
                    previous_factor_embedding_size=8,
                    hidden_size=8,
                    decode_chunk_size=5,
                ),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_backward_smoke(self) -> None:
        model, source, vocabularies = self._model()
        model = model.to("cuda")
        tokens = [tokens_for_query_tables(source.metadata, {"A", "B"}, {"F_A_to_B"})]
        token_ids = encode_tokens_tensor(tokens, vocabularies, device="cuda")
        loss = model(token_ids).sum()
        loss.backward()
        self.assertTrue(all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
