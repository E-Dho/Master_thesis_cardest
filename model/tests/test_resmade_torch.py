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
from model.src.model.anpm import ANPMColumnDecoder, ANPMConfig, GeneratedANPMParameters
from model.src.model.checkpoint import load_resmade_checkpoint, save_resmade_checkpoint
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
    valid_factor_class_mask,
)
from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
from model.src.training.resmade_trainer import (
    build_resmade_from_config,
    train_resmade_sample_source,
)
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.training.torch_losses import torch_weighted_per_head_cross_entropy
from model.src.inference.torch_estimator import TorchDistributionModel
from model.src.model.output_adapter import TorchBackboneOutputs


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

    def test_training_early_stopping_records_stop_reason(self) -> None:
        source = SyntheticFullJoinSampleSource()
        config = load_simple_yaml("model/configs/resmade_smoke.yaml")
        with tempfile.TemporaryDirectory() as output_directory:
            config["model"]["hidden_sizes"] = [16]
            config["model"]["embedding_size"] = 4
            config["training"]["batch_size"] = 4
            config["training"]["steps_per_epoch"] = 10
            config["training"]["checkpoint_interval_steps"] = 0
            config["training"]["validation_interval_steps"] = 1
            config["training"]["early_stopping_patience_steps"] = 1
            config["training"]["early_stopping_min_delta"] = 1.0e9
            config["logging"]["output_directory"] = output_directory

            result = train_resmade_sample_source(source, config)

            early_stopping = result.early_stopping_summary
            self.assertTrue(early_stopping["enabled"])
            self.assertTrue(early_stopping["stopped"])
            self.assertEqual(early_stopping["monitor"], "loss")
            self.assertEqual(early_stopping["patience_steps"], 1)
            self.assertEqual(early_stopping["stop_step"], 2)
            self.assertEqual(result.total_sampled_tuples, 8)

    def test_compositional_predicate_encoding_shares_literal_values(self) -> None:
        source = SyntheticFullJoinSampleSource()
        config = load_simple_yaml("model/configs/resmade_smoke.yaml")
        config["predicate_encoding"] = {
            "mode": "compositional",
            "operator_embedding_size": 4,
            "value_embedding_size": 6,
            "special_embedding_size": 3,
            "merge_hidden_size": 8,
        }
        model = build_resmade_from_config(source.metadata, config)
        vocabularies = PredicateVocabularies.from_metadata(source.metadata)
        equal_id = vocabularies.encode_token(0, PredicateToken.equal("a1"))
        le_id = vocabularies.encode_token(0, PredicateToken(PredicateOp.LESS_EQUAL, "a1"))
        ge_id = vocabularies.encode_token(0, PredicateToken(PredicateOp.GREATER_EQUAL, "a1"))

        value_lookup = model.token_value_ids_0.detach().cpu()
        operator_lookup = model.token_operator_ids_0.detach().cpu()
        self.assertEqual(int(value_lookup[equal_id]), int(value_lookup[le_id]))
        self.assertEqual(int(value_lookup[equal_id]), int(value_lookup[ge_id]))
        self.assertNotEqual(int(operator_lookup[equal_id]), int(operator_lookup[le_id]))
        self.assertNotEqual(int(operator_lookup[equal_id]), int(operator_lookup[ge_id]))

        token_rows = [[PredicateToken.equal("a1")] + [PredicateToken.wildcard()] * (len(source.metadata.columns) - 1)]
        token_ids = encode_tokens_tensor(token_rows, vocabularies)
        logits = model(token_ids)
        logits.sum().backward()
        self.assertGreater(float(model.operator_embedding.weight.grad.norm()), 0.0)
        self.assertGreater(float(model.value_embeddings[0].weight.grad.norm()), 0.0)
        self.assertGreater(float(model.special_embedding.weight.grad.norm()), 0.0)

    def test_hybrid_predicate_encoding_adds_literal_specific_capacity(self) -> None:
        source = SyntheticFullJoinSampleSource()
        config = load_simple_yaml("model/configs/resmade_smoke.yaml")
        config["predicate_encoding"] = {
            "mode": "hybrid",
            "operator_embedding_size": 4,
            "value_embedding_size": 6,
            "special_embedding_size": 3,
            "merge_hidden_size": 8,
        }
        model = build_resmade_from_config(source.metadata, config)
        vocabularies = PredicateVocabularies.from_metadata(source.metadata)
        self.assertEqual(len(model.literal_embeddings), len(source.metadata.columns))

        equal_id = vocabularies.encode_token(0, PredicateToken.equal("a1"))
        le_id = vocabularies.encode_token(0, PredicateToken(PredicateOp.LESS_EQUAL, "a1"))
        value_lookup = model.token_value_ids_0.detach().cpu()
        self.assertEqual(int(value_lookup[equal_id]), int(value_lookup[le_id]))

        token_rows = [[PredicateToken.equal("a1")] + [PredicateToken.wildcard()] * (len(source.metadata.columns) - 1)]
        token_ids = encode_tokens_tensor(token_rows, vocabularies)
        logits = model(token_ids)
        logits.sum().backward()
        self.assertGreater(float(model.literal_embeddings[0].weight.grad[equal_id].norm()), 0.0)
        self.assertGreater(float(model.operator_embedding.weight.grad.norm()), 0.0)
        self.assertGreater(float(model.value_embeddings[0].weight.grad.norm()), 0.0)

    def test_two_slot_native_range_conditions_later_logits(self) -> None:
        source = SyntheticFullJoinSampleSource()
        config = load_simple_yaml("model/configs/resmade_smoke.yaml")
        config["predicate_encoding"] = {
            "mode": "two_slot",
            "operator_embedding_size": 4,
            "value_embedding_size": 6,
            "merge_hidden_size": 8,
        }
        model = build_resmade_from_config(source.metadata, config)
        vocabularies = PredicateVocabularies.from_metadata(
            source.metadata,
            encoding_mode="two_slot",
        )
        wildcard = [PredicateToken.wildcard()] * len(source.metadata.columns)
        ranged = list(wildcard)
        ranged[0] = PredicateToken.range("a1", "a2")
        logits_wildcard = model(encode_tokens_tensor([wildcard], vocabularies))
        logits_range = model(encode_tokens_tensor([ranged], vocabularies))
        later_start, later_stop = source.metadata.model_output_slices[1]
        self.assertFalse(
            torch.allclose(
                logits_wildcard[:, later_start:later_stop],
                logits_range[:, later_start:later_stop],
            )
        )

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

    def test_distjoin_hypernetwork_shapes_and_batch_validation(self) -> None:
        decoder = ANPMColumnDecoder(
            factor_domains=(3, 5, 7),
            embedding_size=4,
            hidden_size=6,
        )
        prefix = torch.tensor([[0], [2]])
        prefix_embedding = decoder._encode_prefix(1, prefix)
        generated = decoder.generated_parameters(1, prefix_embedding)
        self.assertEqual(generated.first_left.shape, (2, 5))
        self.assertEqual(generated.first_right.shape, (2, 6))
        self.assertEqual(generated.hidden_bias.shape, (2, 6))
        self.assertEqual(generated.second_left.shape, (2, 6))
        self.assertEqual(generated.second_right.shape, (2, 5))
        self.assertEqual(generated.logit_bias.shape, (2, 5))
        first_weight, second_weight = reference_low_rank_weight_matrices(generated)
        self.assertEqual(first_weight.shape, (2, 5, 6))
        self.assertEqual(second_weight.shape, (2, 6, 5))

        expanded = decoder.conditional_logits(1, torch.randn(1, 5), prefix)
        self.assertEqual(expanded.shape, (2, 5))
        factor_two = decoder.conditional_logits(2, torch.randn(2, 7), torch.tensor([[0, 1], [2, 4]]))
        self.assertEqual(factor_two.shape, (2, 7))
        with self.assertRaises(ValueError):
            decoder.conditional_logits(1, torch.randn(3, 5), prefix)

    def test_distjoin_transform_matches_reference_equation(self) -> None:
        decoder = ANPMColumnDecoder(
            factor_domains=(3, 5, 7),
            embedding_size=4,
            hidden_size=6,
        )
        for factor_index, domain in [(1, 5), (2, 7)]:
            prefix = torch.randint(0, 3, (2, factor_index))
            if factor_index == 2:
                prefix[:, 1] = torch.randint(0, 5, (2,))
            base_logits = torch.randn(2, domain)
            generated = decoder.generated_parameters(
                factor_index,
                decoder._encode_prefix(factor_index, prefix),
            )
            expected = reference_distjoin_transform(
                base_logits,
                generated,
                final_activation="relu",
            )
            self.assertTrue(torch.allclose(decoder.distjoin_transform(base_logits, generated), expected))

        one_row_prefix = torch.tensor([[1]])
        one_row_base = torch.randn(1, 5)
        one_row_params = decoder.generated_parameters(
            1,
            decoder._encode_prefix(1, one_row_prefix),
        )
        expected = reference_distjoin_transform(one_row_base, one_row_params)
        self.assertTrue(torch.allclose(decoder.distjoin_transform(one_row_base, one_row_params), expected))

    def test_rank_one_contraction_matches_explicit_matrix_gradients(self) -> None:
        for final_activation in ("relu", "identity"):
            torch.manual_seed(11)
            decoder = ANPMColumnDecoder(
                factor_domains=(4, 9),
                embedding_size=5,
                hidden_size=7,
                final_activation=final_activation,
            )
            base = torch.randn(6, 9, requires_grad=True)
            prefix = torch.tensor([[0], [1], [2], [3], [0], [2]])
            parameters = decoder.generated_parameters(1, decoder._encode_prefix(1, prefix))
            optimized = decoder.distjoin_transform(base, parameters)
            reference = reference_distjoin_transform(
                base,
                parameters,
                final_activation=final_activation,
            )
            torch.testing.assert_close(optimized, reference, rtol=1.0e-5, atol=1.0e-6)

            optimized_loss = optimized.square().sum()
            reference_loss = reference.square().sum()
            optimized_grads = torch.autograd.grad(
                optimized_loss,
                [base, *list(decoder.parameters())],
                retain_graph=True,
                allow_unused=True,
            )
            reference_grads = torch.autograd.grad(
                reference_loss,
                [base, *list(decoder.parameters())],
                allow_unused=True,
            )
            for optimized_grad, reference_grad in zip(optimized_grads, reference_grads):
                if optimized_grad is None or reference_grad is None:
                    self.assertIs(optimized_grad, reference_grad)
                    continue
                torch.testing.assert_close(
                    optimized_grad,
                    reference_grad,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )

    def test_hypernetwork_prefix_changes_context_transformation(self) -> None:
        decoder = ANPMColumnDecoder(
            factor_domains=(3, 4),
            embedding_size=2,
            hidden_size=3,
            final_activation="identity",
        )
        with torch.no_grad():
            decoder.previous_factor_embeddings[0].weight.copy_(
                torch.tensor([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
            )
            for parameter in decoder.factor_hypernetworks[0].parameters():
                parameter.fill_(0.2)
        c1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        c2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        p1 = torch.tensor([[0]])
        p2 = torch.tensor([[2]])
        delta = (
            decoder.conditional_logits(1, c1, p1)
            - decoder.conditional_logits(1, c1, p2)
            - decoder.conditional_logits(1, c2, p1)
            + decoder.conditional_logits(1, c2, p2)
        )
        self.assertGreater(float(delta.abs().max()), 1.0e-5)

    def test_anpm_prefix_scope_and_out_of_domain_validation(self) -> None:
        decoder = ANPMColumnDecoder(
            factor_domains=(3, 5, 7),
            embedding_size=4,
            hidden_size=6,
        )
        true_factors = torch.tensor([[1, 0, 0], [1, 4, 6]])
        base_logits = [
            torch.randn(2, 3),
            torch.randn(1, 5).expand(2, -1).clone(),
            torch.randn(1, 7).expand(2, -1).clone(),
        ]
        decoded = decoder.training_logits(base_logits, true_factors)
        self.assertTrue(torch.allclose(decoded[0], base_logits[0]))
        self.assertTrue(torch.allclose(decoded[1][0], decoded[1][1]))
        self.assertFalse(torch.allclose(decoded[2][0], decoded[2][1]))
        with self.assertRaises(ValueError):
            decoder.conditional_logits(1, torch.randn(1, 5), torch.tensor([[3]]))

    def test_hypernetwork_gradients_reach_all_generators(self) -> None:
        decoder = ANPMColumnDecoder(
            factor_domains=(3, 5),
            embedding_size=4,
            hidden_size=6,
        )
        for parameter in decoder.parameters():
            torch.nn.init.constant_(parameter, 0.1)
        base_logits = torch.ones(3, 5, requires_grad=True)
        prefix = torch.tensor([[0], [1], [2]])
        loss = decoder.conditional_logits(1, base_logits, prefix).sum()
        loss.backward()
        self.assertIsNotNone(base_logits.grad)
        self.assertGreater(float(base_logits.grad.abs().sum()), 0.0)
        embedding_grad = decoder.previous_factor_embeddings[0].weight.grad
        self.assertIsNotNone(embedding_grad)
        self.assertGreater(float(embedding_grad.abs().sum()), 0.0)
        hypernetwork = decoder.factor_hypernetworks[0]
        for name, module in [
            ("first_left", hypernetwork.first_left),
            ("first_right", hypernetwork.first_right),
            ("hidden_bias", hypernetwork.hidden_bias),
            ("second_left", hypernetwork.second_left),
            ("second_right", hypernetwork.second_right),
            ("logit_bias", hypernetwork.logit_bias),
        ]:
            grads = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(grad is not None for grad in grads), name)
            self.assertTrue(all(torch.all(torch.isfinite(grad)) for grad in grads), name)
            self.assertGreater(sum(float(grad.abs().sum()) for grad in grads), 0.0, name)

    def test_valid_factor_masks_zero_invalid_probabilities(self) -> None:
        metadata = self._factorized_metadata()
        factorization = metadata.factorization_plan.original_column_factorizations[0]
        decoder = ANPMColumnDecoder(
            factor_domains=factorization.factor_domains,
            embedding_size=4,
            hidden_size=6,
        )
        prefix0 = torch.empty(1, 0, dtype=torch.long)
        mask0 = valid_factor_class_mask(factorization, 0, prefix0)
        logits0 = decoder.conditional_logits(
            0,
            torch.zeros(1, factorization.factor_domains[0]),
            prefix0,
            valid_class_mask=mask0,
        )
        probs0 = torch.softmax(logits0, dim=1)
        self.assertTrue(torch.all(probs0[0, ~mask0[0]] == 0.0))
        self.assertAlmostEqual(float(probs0.sum()), 1.0, places=6)

        prefix1 = torch.tensor([[2]])
        mask1 = valid_factor_class_mask(factorization, 1, prefix1)
        self.assertEqual(mask1.tolist(), [[True, True, True, True, False, False, False, False]])
        logits1 = decoder.conditional_logits(
            1,
            torch.zeros(1, factorization.factor_domains[1]),
            prefix1,
            valid_class_mask=mask1,
        )
        probs1 = torch.softmax(logits1, dim=1)
        self.assertTrue(torch.all(probs1[0, ~mask1[0]] == 0.0))
        self.assertAlmostEqual(float(probs1.sum()), 1.0, places=6)
        with self.assertRaises(ValueError):
            decoder.conditional_logits(
                1,
                torch.zeros(1, factorization.factor_domains[1]),
                prefix1,
                valid_class_mask=torch.zeros_like(mask1),
            )

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

    def test_factorized_equality_and_range_fast_paths_match_distribution(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        wrapped = TorchDistributionModel(model, metadata, vocabularies)
        wildcard_tokens = [PredicateToken.wildcard(), PredicateToken.wildcard()]
        distribution = wrapped.predict_distributions(wildcard_tokens)[0]
        token_ids = encode_tokens_tensor([wildcard_tokens], vocabularies)
        logits = model(token_ids)
        outputs = TorchBackboneOutputs(
            logits=logits,
            split_logits=model.split_logits(logits),
        )

        def direct_factor(token: PredicateToken) -> float:
            factor = wrapped.output_adapter.column_factor(
                original_column_index=0,
                backbone_outputs=outputs,
                predicate_token=token,
            )
            return float(factor[0].detach().cpu())

        for token, expected in [
            (PredicateToken.equal(7), distribution[7]),
            (PredicateToken(PredicateOp.LESS_EQUAL, 7), distribution[:8].sum()),
            (PredicateToken(PredicateOp.LESS_THAN, 7), distribution[:7].sum()),
            (PredicateToken(PredicateOp.GREATER_EQUAL, 7), distribution[7:].sum()),
            (PredicateToken(PredicateOp.GREATER_THAN, 7), distribution[8:].sum()),
        ]:
            self.assertAlmostEqual(direct_factor(token), float(expected), places=5)

        self.assertAlmostEqual(
            direct_factor(PredicateToken.range(5, 12)),
            float(distribution[5:13].sum()),
            places=5,
        )

    def test_native_interval_mass_uses_one_backbone_state(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        wrapped = TorchDistributionModel(model, metadata, vocabularies)
        tokens = [PredicateToken.wildcard(), PredicateToken.wildcard()]
        token_ids = encode_tokens_tensor([tokens], vocabularies)
        logits = model(token_ids)
        outputs = TorchBackboneOutputs(
            logits=logits,
            split_logits=model.split_logits(logits),
        )
        before = model.forward_calls
        mass = wrapped.output_adapter.interval_mass(
            original_column_index=0,
            backbone_outputs=outputs,
            lower_literal=5,
            upper_literal=12,
            lower_inclusive=False,
            upper_inclusive=True,
        )
        after = model.forward_calls
        distribution = wrapped.predict_distributions(tokens)[0]
        self.assertEqual(after, before)
        self.assertGreater(float(mass[0]), 0.0)
        self.assertLessEqual(float(mass[0]), 1.0)
        self.assertAlmostEqual(float(mass[0]), float(distribution[6:13].sum()), places=5)

    def test_chunked_and_unchunked_factorized_inference_agree(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        tokens = [PredicateToken.wildcard(), PredicateToken.wildcard()]
        wrapped = TorchDistributionModel(model, metadata, vocabularies)
        chunked = wrapped.predict_distributions(tokens)[0]
        wrapped.output_adapter.anpm_config = ANPMConfig(
            enabled=True,
            previous_factor_embedding_size=8,
            hidden_size=8,
            decode_chunk_size=100,
        )
        unchunked = wrapped.predict_distributions(tokens)[0]
        self.assertTrue(np.allclose(chunked, unchunked, atol=1.0e-6))

    def test_factorized_training_loss_is_finite_with_validity_masks(self) -> None:
        metadata = self._factorized_metadata()
        vocabularies = PredicateVocabularies.from_metadata(metadata)
        model = self._factorized_model(metadata, vocabularies)
        rows = torch.tensor([[0, 0], [19, 1], [7, 0], [13, 1]])
        weights = torch.ones_like(rows, dtype=torch.float32)
        tokens = [[PredicateToken.wildcard(), PredicateToken.wildcard()]]
        logits = model(encode_tokens_tensor(tokens, vocabularies).expand(rows.shape[0], -1))
        breakdown = torch_weighted_per_head_cross_entropy(
            logits,
            rows,
            weights,
            metadata,
            anpm_decoders=model.anpm_decoders,
        )
        self.assertTrue(torch.isfinite(breakdown.total_loss))


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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_backward_smoke(self) -> None:
        model, source, vocabularies = self._model()
        model = model.to("cuda")
        tokens = [tokens_for_query_tables(source.metadata, {"A", "B"}, {"F_A_to_B"})]
        token_ids = encode_tokens_tensor(tokens, vocabularies, device="cuda")
        loss = model(token_ids).sum()
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
                for parameter in model.parameters()
            )
        )

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

def reference_distjoin_transform(
    base_logits: torch.Tensor,
    parameters: GeneratedANPMParameters,
    *,
    final_activation: str = "relu",
) -> torch.Tensor:
    """Direct DistJoin ANPM equation used to test the production decoder."""

    first_weight, second_weight = reference_low_rank_weight_matrices(parameters)
    hidden = torch.relu(
        torch.bmm(base_logits.unsqueeze(1), first_weight)
        + parameters.hidden_bias.unsqueeze(1)
    )
    logits = (
        torch.bmm(hidden, second_weight)
        + parameters.logit_bias.unsqueeze(1)
    ).squeeze(1)
    if final_activation == "relu":
        return torch.relu(logits)
    if final_activation == "identity":
        return logits
    raise ValueError(final_activation)


def reference_low_rank_weight_matrices(
    parameters: GeneratedANPMParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-only explicit W1/W2 materialization for parity checks."""

    first_weight = torch.bmm(
        parameters.first_left.unsqueeze(-1),
        parameters.first_right.unsqueeze(1),
    )
    second_weight = torch.bmm(
        parameters.second_left.unsqueeze(-1),
        parameters.second_right.unsqueeze(1),
    )
    return first_weight, second_weight


if __name__ == "__main__":
    unittest.main()
