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
from model.src.model.checkpoint import load_resmade_checkpoint, save_resmade_checkpoint
from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.training.torch_losses import torch_weighted_per_head_cross_entropy


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

