from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from Application.model import BPRMF as bpr
from Application.model.training_data import Mappings, Splits, PreparedBprData, PreparedDataError, validate_prepared_data
from Application.model.bpr_preparation import BprMappings, BprSplits, prepare_bpr


@pytest.fixture
def prepared():
    return PreparedBprData(
        Mappings({"u0": 0, "u1": 1}, ["u0", "u1"], {"i0": 0, "i1": 1, "i2": 2}, ["i0", "i1", "i2"]),
        Splits(np.array([[0, 0], [1, 1]]), np.array([0.1, 10.0]), np.array([0]), np.array([2]), [{0}, {1}]),
    )


def test_shared_reexports_and_source_neutral_result():
    assert bpr.Mappings is BprMappings is Mappings
    assert bpr.Splits is BprSplits is Splits
    assert isinstance(prepare_bpr([]), PreparedBprData)


def test_controlled_mutability_is_explicit_and_revalidated(prepared):
    validate_prepared_data(prepared)
    weights = prepared.splits.train_weights
    alias = PreparedBprData(prepared.mappings, prepared.splits)
    weights[0] = 0
    assert alias.splits.train_weights is weights
    validate_prepared_data(alias)  # zero weight is legal in legacy loss
    weights[0] = np.nan
    with pytest.raises(PreparedDataError):
        validate_prepared_data(alias)
    assert "u0" not in repr(prepared) + repr(prepared.mappings) + repr(prepared.splits)


@pytest.mark.parametrize("field,value", [
    ("train_pairs", np.array([[2, 0]])), ("train_pairs", np.array([[0, 3], [1, 1]])),
    ("train_pairs", np.array([[-1, 0], [1, 1]])), ("train_pairs", np.array([[0., 0.], [1., 1.]])),
    ("train_pairs", np.array([0, 1])), ("train_pairs", np.zeros((2, 3), dtype=int)),
    ("train_weights", np.array([1.])), ("train_weights", np.array([np.nan, 1.])),
    ("train_weights", np.array([np.inf, 1.])), ("train_weights", np.array([-1., 1.])),
    ("train_weights", np.array([[1., 1.]])), ("eval_users", np.array([2])),
    ("eval_items", np.array([3])), ("eval_items", np.array([-1])),
    ("eval_items", np.array([1, 2])), ("eval_users", np.array([0.])),
    ("user_pos_train", [{0}]), ("user_pos_train", [{2}, {1}]),
])
def test_invalid_splits(prepared, field, value):
    broken = replace(prepared, splits=replace(prepared.splits, **{field: value}))
    with pytest.raises(PreparedDataError):
        validate_prepared_data(broken)


@pytest.mark.parametrize("field,value", [
    ("user2idx", {"u0": 1, "u1": 0}), ("user2idx", {"u0": 0}),
    ("idx2user", ["u0", "u0"]), ("item2idx", {"i0": 0, "i1": 1, "i2": 9}),
    ("idx2item", ["i0", "i1"]), ("user2idx", {"u0": False, "u1": 1}),
])
def test_invalid_mappings(prepared, field, value):
    with pytest.raises(PreparedDataError):
        validate_prepared_data(replace(prepared, mappings=replace(prepared.mappings, **{field: value})))


def test_empty_and_wrong_input():
    for data in (None, object(), prepare_bpr([])):
        with pytest.raises(PreparedDataError):
            validate_prepared_data(data)


def write_csv(tmp_path):
    frames = [pd.DataFrame({"MindboxID": ["u0", "u1"], "КодНоменклатуры": ["i0", "i1"],
                            "Количество": ["1.5", "2"], "Дата": ["2026-01-01", "2026-01-02"]}),
              pd.DataFrame({"MindboxID": ["u0"], "КодНоменклатуры": ["i2"],
                            "ТипТовара": ["Номенклатура"], "Дата": ["2026-01-03"]}),
              pd.DataFrame(columns=["MindboxID", "КодНоменклатуры", "Дата"])]
    for name, frame in zip(("orders", "views", "favorites"), frames):
        frame.to_csv(tmp_path / f"{name}.csv", sep="|", encoding="utf-8-sig", index=False)
    return frames


def assert_splits_equal(left, right):
    for name in ("train_pairs", "train_weights", "eval_users", "eval_items"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert left.user_pos_train == right.user_pos_train


def test_csv_preparation_and_public_entrypoint_parity(tmp_path, monkeypatch):
    frames = write_csv(tmp_path)
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), min_user_interactions_for_eval=2)
    maps = bpr._build_mappings(*frames)
    events = bpr._collect_user_item_events(*frames, maps, cfg)
    expected = bpr._train_test_split_last_per_user(events, cfg, len(maps.idx2user))
    data = bpr.prepare_training_data_from_csv(cfg)
    assert data.mappings == maps
    assert_splits_equal(data.splits, expected)
    calls = []
    real_prepare = bpr.prepare_training_data_from_csv

    def prepare(config):
        calls.append("prepare")
        return real_prepare(config)

    def train(config, actual, device):
        calls.append("train")
        assert actual.mappings == maps
        assert_splits_equal(actual.splits, expected)
        return object(), actual.splits

    monkeypatch.setattr(bpr, "prepare_training_data_from_csv", prepare)
    monkeypatch.setattr(bpr, "train_prepared_data", train)
    monkeypatch.setattr(bpr, "_save_artifacts", lambda *args, **kwargs: calls.append("save"))
    assert bpr._train_in_this_process(cfg)
    assert calls == ["prepare", "train", "save"]


@pytest.mark.parametrize("features", [False, True])
def test_prepared_core_without_interaction_csv(tmp_path, monkeypatch, prepared, features):
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), use_item_features=features, epochs=1,
                          embedding_dim=4, batch_size=2, n_neg=1, topk=2)
    if features:
        pd.DataFrame({"КодНоменклатуры": ["i0", "i1", "i2"], "Марка": ["a", "b", "a"]}).to_csv(
            tmp_path / "nomenclature.csv", sep="|", index=False, encoding="utf-8-sig")
    reads = []
    original = bpr._read_csv_pipe

    def read(path):
        assert str(path).endswith("nomenclature.csv")
        reads.append(path)
        return original(path)

    def forbidden(*args, **kwargs):
        pytest.fail("Prepared core must not prepare interaction sources")

    monkeypatch.setattr(bpr, "_read_csv_pipe", read)
    for name in ("_require_interaction_sources", "_validate_interaction_source_schemas",
                 "_collect_user_item_events", "_train_test_split_last_per_user", "_save_artifacts"):
        monkeypatch.setattr(bpr, name, forbidden)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    bpr._set_seed(cfg.seed)
    model, splits = bpr.train_prepared_data(cfg, prepared, torch.device("cpu"))
    assert splits is prepared.splits
    assert bool(reads) == features
    assert model.use_item_features == features
    assert len(list(tmp_path.iterdir())) == int(features)


def test_short_training_legacy_wrapper_vs_prepared(tmp_path, monkeypatch):
    frames = write_csv(tmp_path)
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), use_item_features=False, epochs=2,
                          embedding_dim=4, batch_size=3, n_neg=1, topk=2)
    maps = bpr._build_mappings(*frames)
    events = bpr._collect_user_item_events(*frames, maps, cfg)
    data = bpr.prepare_training_data_from_csv(cfg)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    bpr._set_seed(cfg.seed)
    old_model, old_split = bpr.train_bprmf(maps, events, cfg, torch.device("cpu"))
    bpr._set_seed(cfg.seed)
    new_model, new_split = bpr.train_prepared_data(cfg, data, torch.device("cpu"))
    assert_splits_equal(old_split, new_split)
    for name, value in old_model.state_dict().items():
        torch.testing.assert_close(value, new_model.state_dict()[name])
    assert not (tmp_path / "model").exists()


def test_validation_precedes_training_side_effects(prepared, monkeypatch):
    prepared.splits.train_weights[0] = np.inf
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: pytest.fail("Must validate first"))
    with pytest.raises(PreparedDataError):
        bpr.train_prepared_data(bpr.TrainConfig(), prepared, torch.device("cpu"))


@pytest.mark.parametrize("weights", [
    np.array([1., 10., 100., 2., 0., 50., 3.]),
    np.zeros(7),
])
def test_positive_selection_is_independent_of_weights(weights):
    pairs = np.column_stack((np.arange(7), np.arange(7) + 20))
    original_pairs, original_weights = pairs.copy(), weights.copy()
    uniform_rng, weighted_rng = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(100):
        uniform_users, uniform_items, _ = bpr._sample_batch(pairs, np.ones(7), 3, uniform_rng)
        users, items, batch_weights = bpr._sample_batch(pairs, weights, 3, weighted_rng)
        np.testing.assert_array_equal(users, uniform_users)
        np.testing.assert_array_equal(items, uniform_items)
        np.testing.assert_array_equal(batch_weights, weights[users])
        assert len(set(users)) == 3  # Preserve sampling without replacement within a batch.
        assert batch_weights.dtype == np.float64
    np.testing.assert_array_equal(pairs, original_pairs)
    np.testing.assert_array_equal(weights, original_weights)


def test_positive_selection_is_uniform_for_unequal_weights():
    pairs = np.array([[0, 0], [1, 1]])
    weights = np.array([1., 10.])
    rng = np.random.default_rng(42)
    counts = np.zeros(2, dtype=int)
    for _ in range(20000):
        users, _, batch_weights = bpr._sample_batch(pairs, weights, 1, rng)
        counts[users[0]] += 1
        assert batch_weights[0] == weights[users[0]]
    np.testing.assert_allclose(counts / counts.sum(), [0.5, 0.5], atol=0.015, rtol=0)


@pytest.mark.parametrize("batch_size", [2, 5])
def test_full_batch_preserves_pairs_weights_and_rng_state(batch_size):
    pairs = np.array([[3, 8], [1, 5]])
    weights = np.array([1, 10], dtype=np.int32)
    rng = np.random.default_rng(42)
    state = rng.bit_generator.state
    users, items, batch_weights = bpr._sample_batch(pairs, weights, batch_size, rng)
    np.testing.assert_array_equal(users, pairs[:, 0])
    np.testing.assert_array_equal(items, pairs[:, 1])
    np.testing.assert_array_equal(batch_weights, weights)
    assert batch_weights.dtype == np.float64
    assert rng.bit_generator.state == state


class _FixedScoreModel(torch.nn.Module):
    """Known logits for checking the production loop's loss and gradients."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.user_emb = torch.nn.Embedding(2, 1)
        self.item_emb = torch.nn.Embedding(3, 1)
        self.use_item_features = False
        with torch.no_grad():
            self.user_emb.weight.fill_(1)
            self.item_emb.weight.copy_(torch.tensor([[0.], [2.], [1.]]))

    def score(self, users, items):
        user_vectors = self.user_emb(users)
        if items.ndim == 2:
            user_vectors = user_vectors.unsqueeze(1)
        return (user_vectors * self.item_emb(items)).sum(dim=-1)


@pytest.mark.parametrize("weights", [[1., 10.], [10., 1.], [1., 1.], [0., 10.]])
@pytest.mark.parametrize("bpr_reg", [0., 0.25])
def test_training_retains_normalized_weighted_loss_and_gradients(tmp_path, monkeypatch, weights, bpr_reg):
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), epochs=1, embedding_dim=1,
                           batch_size=2, n_neg=2, use_item_features=False, lr=0., bpr_reg=bpr_reg)
    prepared = PreparedBprData(
        Mappings({"u0": 0, "u1": 1}, ["u0", "u1"],
                 {"i0": 0, "i1": 1, "i2": 2}, ["i0", "i1", "i2"]),
        Splits(np.array([[0, 0], [1, 1]]), np.array(weights),
               np.array([], dtype=int), np.array([], dtype=int), [{0}, {1}]),
    )
    negatives = np.array([[2, 1], [2, 0]])
    def fixed_negatives(users, num_items, positives, rng, n_neg):
        np.testing.assert_array_equal(users, [0, 1])
        assert num_items == 3 and positives == [{0}, {1}] and n_neg == 2
        return negatives
    monkeypatch.setattr(bpr, "BPRMF", _FixedScoreModel)
    monkeypatch.setattr(bpr, "_sample_negatives", fixed_negatives)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    model, splits, metrics = bpr.train_prepared_data_with_metrics(cfg, prepared, torch.device("cpu"))
    assert splits is prepared.splits
    np.testing.assert_array_equal(splits.train_weights, weights)

    # Two negatives per pair: deltas [-1, -2] and [1, 2].
    # Compute the reference via softplus, independently of the loop's logsigmoid.
    users, positives = torch.tensor([0, 1]), torch.tensor([0, 1])
    negatives_t = torch.tensor(negatives)
    delta = model.score(users, positives).unsqueeze(1) - model.score(users, negatives_t)
    per_pair = F.softplus(-delta).mean(dim=1)
    weight_t = torch.tensor(weights)
    expected = (per_pair * weight_t).sum() / weight_t.sum().clamp_min(1e-6)
    if bpr_reg:
        reg = (model.user_emb(users).pow(2).sum(dim=-1)
               + model.item_emb(positives).pow(2).sum(dim=-1)
               + model.item_emb(negatives_t).pow(2).sum(dim=-1).mean(dim=1)).mean()
        expected = expected + bpr_reg * reg
    assert metrics.history[0].loss == pytest.approx(expected.item())
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
    for parameter, expected_grad in zip(model.parameters(), expected_grads):
        torch.testing.assert_close(parameter.grad, expected_grad)


def _assert_valid_negatives(users, negatives, num_items, positives):
    assert negatives.dtype == np.int64
    assert (negatives >= 0).all() and (negatives < num_items).all()
    for user, row in zip(users, negatives):
        assert set(row).isdisjoint(positives[int(user)])


@pytest.mark.parametrize("n_neg", [1, 7])
def test_negative_sampling_sparse_multiple_users_bounds_and_unchanged_inputs(n_neg):
    users = np.array([0, 1, 0, 2, 3])
    positives = [{0, 3, 9}, {2, 4}, set(), {1, 7, 11}]
    original_users, original_positives = users.copy(), [set(items) for items in positives]
    negatives = bpr._sample_negatives(users, 100, positives, np.random.default_rng(42), n_neg=n_neg)
    assert negatives.shape == (len(users), n_neg)
    _assert_valid_negatives(users, negatives, 100, positives)
    np.testing.assert_array_equal(users, original_users)
    assert positives == original_positives


def test_negative_sampling_last_retry_is_checked():
    users, positives = np.array([0]), [{0, 1, 2, 3}]
    negatives = bpr._sample_negatives(users, 5, positives, np.random.default_rng(0), n_neg=10, max_tries=1)
    # Old implementation deterministically returned six positives in this batch.
    np.testing.assert_array_equal(negatives, np.full((1, 10), 4))


class _CollisionRng:
    """Force rejection to exhaust retries, then use real uniform fallback draws."""

    def __init__(self, seed=42):
        self.generator = np.random.default_rng(seed)
        self.integer_calls = 0
        self.choice_calls = []

    def integers(self, low, high, size, dtype):
        self.integer_calls += 1
        return np.zeros(size, dtype=dtype)

    def choice(self, candidates, size, replace):
        self.choice_calls.append((candidates.copy(), size, replace))
        return self.generator.choice(candidates, size=size, replace=replace)


@pytest.mark.parametrize("max_tries", [0, 2])
def test_negative_sampling_forced_dense_fallback_and_duplicates(max_tries):
    users, positives = np.array([0, 0]), [{0, 1, 2, 3}]
    rng = _CollisionRng()
    negatives = bpr._sample_negatives(users, 5, positives, rng, n_neg=7, max_tries=max_tries)
    np.testing.assert_array_equal(negatives, np.full((2, 7), 4))
    assert rng.integer_calls == 1 + max_tries
    assert rng.choice_calls and all(call[2] is True for call in rng.choice_calls)
    assert positives == [{0, 1, 2, 3}]
    np.testing.assert_array_equal(users, [0, 0])


def test_negative_sampling_fallback_uniform_seeded_and_reuses_user_complement(monkeypatch):
    users, positives = np.array([0, 1, 0]), [{0, 1, 3}, {0, 2, 4}]
    calls = []
    original = np.setdiff1d
    def track_complement(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "setdiff1d", track_complement)
    negatives = bpr._sample_negatives(users, 5, positives, _CollisionRng(42), n_neg=5000, max_tries=0)
    assert len(calls) == 2  # One complement per unresolved user, even for repeated rows.
    _assert_valid_negatives(users, negatives, 5, positives)
    for user in (0, 1):
        values, counts = np.unique(negatives[users == user], return_counts=True)
        assert set(values) == set(range(5)) - positives[user]
        np.testing.assert_allclose(counts / counts.sum(), [0.5, 0.5], atol=0.025, rtol=0)
    repeated = bpr._sample_negatives(users, 5, positives, _CollisionRng(42), n_neg=5000, max_tries=0)
    np.testing.assert_array_equal(negatives, repeated)


@pytest.mark.parametrize("max_tries", [0, 1, 25])
def test_negative_sampling_no_unseen_fails_before_draws(max_tries):
    users, positives = np.array([0, 1]), [{0}, set(range(5))]
    rng = _CollisionRng()
    with pytest.raises(ValueError, match="No candidate negative items") as caught:
        bpr._sample_negatives(users, 5, positives, rng, n_neg=4, max_tries=max_tries)
    assert str(caught.value) == "No candidate negative items for a training user"
    assert rng.integer_calls == 0 and rng.choice_calls == []
    np.testing.assert_array_equal(users, [0, 1])
    assert positives == [{0}, set(range(5))]


@pytest.mark.parametrize("max_tries", [0, 1, 25])
def test_negative_sampling_reproducible_with_numpy_generator(max_tries):
    users, positives = np.array([0, 1, 0]), [{0, 1, 2, 3}, {1, 4}]
    first = bpr._sample_negatives(users, 5, positives, np.random.default_rng(42), n_neg=20, max_tries=max_tries)
    second = bpr._sample_negatives(users, 5, positives, np.random.default_rng(42), n_neg=20, max_tries=max_tries)
    np.testing.assert_array_equal(first, second)
    _assert_valid_negatives(users, first, 5, positives)


def test_negative_sampling_fast_path_does_not_build_complements(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Sparse fast path must not build item complements")
    monkeypatch.setattr(np, "setdiff1d", forbidden)
    users, positives = np.array([0, 1]), [{0, 1}, {2, 3}]
    negatives = bpr._sample_negatives(users, 1000, positives, np.random.default_rng(42), n_neg=10)
    _assert_valid_negatives(users, negatives, 1000, positives)


def test_negative_sampling_fallback_preserves_already_valid_draws():
    class MixedRng(_CollisionRng):
        def integers(self, low, high, size, dtype):
            self.integer_calls += 1
            return np.array([[0, 4, 0, 4]], dtype=dtype)
    rng = MixedRng()
    negatives = bpr._sample_negatives(np.array([0]), 5, [{0, 1, 2}], rng, n_neg=4, max_tries=0)
    np.testing.assert_array_equal(negatives[0, [1, 3]], [4, 4])
    _assert_valid_negatives([0], negatives, 5, [{0, 1, 2}])
    assert len(rng.choice_calls) == 1 and rng.choice_calls[0][1] == 2


def test_negative_sampling_empty_batch_and_no_positive_user():
    rng = _CollisionRng()
    empty = bpr._sample_negatives(np.array([], dtype=int), 0, [], rng, n_neg=3)
    assert empty.shape == (0, 3) and empty.dtype == np.int64 and rng.integer_calls == 0
    negatives = bpr._sample_negatives(np.array([0]), 1, [set()], np.random.default_rng(42), n_neg=3)
    np.testing.assert_array_equal(negatives, [[0, 0, 0]])


def test_training_comparisons_use_train_positives_and_valid_negatives(tmp_path, monkeypatch):
    pairs = np.array([[0, 0], [0, 1], [1, 1]])
    prepared = PreparedBprData(
        Mappings({"u0": 0, "u1": 1}, ["u0", "u1"],
                 {"i0": 0, "i1": 1, "i2": 2}, ["i0", "i1", "i2"]),
        Splits(pairs, np.array([0.1, 2., 10.]), np.array([], dtype=int), np.array([], dtype=int), [{0, 1}, {1}]),
    )
    comparisons = []
    sampler = bpr._sample_negatives
    def checked(users, num_items, positives, rng, n_neg):
        result = sampler(users, num_items, positives, rng, n_neg=n_neg, max_tries=0)
        _assert_valid_negatives(users, result, num_items, positives)
        comparisons.extend(zip(users, result))
        return result
    monkeypatch.setattr(bpr, "_sample_negatives", checked)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    cfg = bpr.TrainConfig(data_dir=str(tmp_path), use_item_features=False, epochs=1,
                         embedding_dim=4, batch_size=2, n_neg=3, topk=2)
    bpr.train_prepared_data_with_metrics(cfg, prepared, torch.device("cpu"))
    assert len(comparisons) == 4
    assert all(item in prepared.splits.user_pos_train[user] for user, item in pairs)


@pytest.mark.parametrize("consumer", ["validator", "trainer", "evaluator"])
def test_seen_eval_target_fails_before_training_or_scoring(prepared, monkeypatch, consumer):
    prepared.splits.eval_items[0] = 0
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: pytest.fail("Must validate before training"))
    with pytest.raises(PreparedDataError, match="Evaluation targets must be unseen in training") as caught:
        if consumer == "validator":
            validate_prepared_data(prepared)
        elif consumer == "trainer":
            bpr.train_prepared_data(bpr.TrainConfig(), prepared, torch.device("cpu"))
        else:
            bpr._eval_bprmf_recall_ndcg(object(), prepared.splits, 3, 1, torch.device("cpu"))
    assert str(caught.value) == "Evaluation targets must be unseen in training"


@pytest.mark.parametrize("k,expected", [(1, (0., 0.)), (2, (1., 1 / np.log2(3))), (10, (1., 1 / np.log2(3)))])
def test_evaluator_masks_all_train_positives_and_keeps_metric_formulas(k, expected, monkeypatch):
    model = bpr.BPRMF(1, 4, 1)
    with torch.no_grad():
        model.user_emb.weight.fill_(1)
        # Seen item ranks highest; unseen scores are below the old -1e9 masking sentinel.
        model.item_emb.weight.copy_(torch.tensor([[100.], [-3e9], [-2e9], [90.]]))
    splits = Splits(np.array([[0, 0], [0, 3]]), np.ones(2), np.array([0]), np.array([1]), [{0, 3}])
    original, calls = torch.topk, []
    def checked(scores, *args, **kwargs):
        assert torch.isneginf(scores[0, [0, 3]]).all()
        calls.append(True)
        return original(scores, *args, **kwargs)
    monkeypatch.setattr(torch, "topk", checked)
    result = bpr._eval_bprmf_recall_ndcg(model, splits, 4, k, torch.device("cpu"))
    assert result == pytest.approx(expected)
    assert calls == [True]


def test_evaluator_empty_eval_does_not_score(prepared):
    prepared.splits.eval_users = prepared.splits.eval_items = np.array([], dtype=np.int64)
    assert bpr._eval_bprmf_recall_ndcg(object(), prepared.splits, 3, 2, torch.device("cpu")) == (0., 0.)
