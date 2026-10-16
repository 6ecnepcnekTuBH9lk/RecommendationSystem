from threading import Event
from unittest.mock import Mock

import pytest

from Application.customer_card_data import CustomerCardService, DisplayField, SectionContent, ViewState
from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus as S
from Application.model.raw_bpr_inference import RawBprRecommendation, RawBprResult, RawBprStatus


def identity(status=S.MATCHED_VIA_MERGE):
    if status in (S.MATCHED_EXACT, S.MATCHED_VIA_MERGE):
        return CustomerModelIdentityResult("C", "C" if status is S.MATCHED_EXACT else "A", status, 1, "g", "r", "v")
    if status is S.AMBIGUOUS:
        return CustomerModelIdentityResult("C", None, status, 3)
    if status is S.NOT_IN_MODEL:
        return CustomerModelIdentityResult("unknown", None, status, 0)
    return CustomerModelIdentityResult(None, None, status, 0)


@pytest.fixture
def service(tmp_path):
    bridge = Mock()
    bridge.resolve.return_value = identity()
    bridge.is_current.return_value = True
    inference = Mock()
    inference.get_recommendations.return_value = RawBprResult("g", "A", RawBprStatus.OK, tuple(
        RawBprRecommendation(i, f"ORIGINAL-{i}", 1.123456789 / i) for i in range(1, 11)))
    return CustomerCardService(bridge=bridge, inference=inference, model_dir=tmp_path / "model",
                               raw_root=tmp_path / "raw", catalog_path=tmp_path / "catalog.csv")


def test_direct_raw_result_without_catalog_preserves_every_code_rank_score(service):
    result = service.load("current-id")
    assert result.state is ViewState.READY
    expected = service.inference.get_recommendations.return_value.recommendations
    assert tuple(p.recommendation for p in result.products) == expected
    assert all(p.reference_message == "Справочник товаров недоступен" for p in result.products)
    service.inference.get_recommendations.assert_called_once_with("A", k=10, model_dir=service.model_dir)
    service.bridge.resolve.assert_called_once_with("current-id", model_dir=service.model_dir, raw_root=service.raw_root)


def test_catalog_enrichment_cannot_filter_replace_or_rerank(service):
    service.catalog_path.write_text(
        "КодНоменклатуры|НазваниеНаСайте|Номенклатура|ТитульнаяФотография|Коллекция|Остаток|ВидНоменклатуры\n"
        "ORIGINAL-1|  |Синтетический товар|/upload/photo.jpg|Осень-Зима 2025|0|Подарочные карты\n",
        encoding="utf-8-sig")
    result = service.load("C")
    assert len(result.products) == 10
    assert result.products[0].name == "Синтетический товар"
    assert result.products[0].photo_url == "/upload/photo.jpg"
    assert result.products[0].collection == "Осень-Зима 2025"
    assert result.products[0].recommendation.model_item_code == "ORIGINAL-1"
    assert result.products[1].reference_message == "Нет в текущем каталоге"
    assert tuple(p.recommendation for p in result.products) == service.inference.get_recommendations.return_value.recommendations


def test_catalog_cache_and_invalidation(service, monkeypatch):
    service.catalog_path.write_text("КодНоменклатуры|Номенклатура\nORIGINAL-1|Первое название\n", encoding="utf-8")
    service.load("C")
    before = service._catalog
    service.load("other-id")
    assert service._catalog is before
    service.catalog_path.write_text("КодНоменклатуры|Номенклатура\nORIGINAL-1|Обновлённое название товара\n", encoding="utf-8")
    assert service.load("C").products[0].name == "Обновлённое название товара"


@pytest.mark.parametrize("status", [S.AMBIGUOUS, S.NOT_IN_MODEL, S.INVALID_ID, S.SOURCE_UNAVAILABLE, S.SOURCE_INCONSISTENT])
def test_unmatched_identity_never_invokes_inference(service, status):
    service.bridge.resolve.return_value = identity(status)
    result = service.load("C")
    assert result.state in (ViewState.ERROR, ViewState.UNAVAILABLE)
    assert result.products == ()
    service.inference.get_recommendations.assert_not_called()


def test_generation_mismatch_and_changed_merges_block_results(service):
    raw = service.inference.get_recommendations.return_value
    service.inference.get_recommendations.return_value = RawBprResult("new", "A", raw.status, raw.recommendations)
    assert service.load("C").state is ViewState.STALE
    service.inference.get_recommendations.return_value = raw
    service.bridge.is_current.return_value = False
    result = service.load("C")
    assert result.state is ViewState.STALE and result.products == ()


def test_model_exception_does_not_leak_internal_details(service):
    service.inference.get_recommendations.side_effect = ValueError("PRIVATE_ID PRIVATE_PATH traceback")
    result = service.load("C")
    assert result.state is ViewState.ERROR
    assert "PRIVATE" not in result.message + repr(result)


def test_bad_catalog_is_reference_error_not_inference_error(service):
    service.catalog_path.write_text("invalid schema", encoding="utf-8")
    result = service.load("C")
    assert result.state is ViewState.READY and len(result.products) == 10
    assert all(p.name == "" and p.reference_message for p in result.products)


def test_cooperative_cancellation_between_stages(service):
    event = Event()
    def resolve(*args, **kwargs):
        event.set()
        return identity()
    service.bridge.resolve.side_effect = resolve
    assert service.load("C", event.is_set).state is ViewState.CANCELLED
    service.inference.get_recommendations.assert_not_called()


def test_empty_result_and_unknown_raw_user_are_explicit(service):
    service.inference.get_recommendations.return_value = RawBprResult("g", "A", RawBprStatus.OK, ())
    assert service.load("C").state is ViewState.EMPTY
    service.inference.get_recommendations.return_value = RawBprResult("g", "A", RawBprStatus.USER_NOT_IN_MODEL, ())
    assert service.load("C").state is ViewState.UNAVAILABLE


def test_future_display_contract_distinguishes_missing_source_value_and_error():
    from Application.tabs.customer_card_widgets import _FIELD_STATES
    assert len({_FIELD_STATES[s] for s in (ViewState.EMPTY, ViewState.NOT_CONNECTED, ViewState.ERROR)}) == 3
    with pytest.raises(ValueError):
        DisplayField("Сумма покупок", "0", ViewState.NOT_CONNECTED)
    with pytest.raises(TypeError):
        SectionContent(ViewState.READY, "Источник подключён", [])


@pytest.mark.parametrize("rows,expected", [
    ("ORIGINAL-1|Имя|Весна\n", "Весна"),
    ("ORIGINAL-1|Имя|\n", ""),
    ("ORIGINAL-1|Имя|Весна\nORIGINAL-1|Имя|Осень\n", ""),
    ("ORIGINAL-1|Первый|Весна\nORIGINAL-1|Другой|Весна\n", ""),
    ("ORIGIN|Другой товар|Весна\n", ""),
])
def test_recommendation_collection_uses_shared_catalog_exact_code_without_reranking(service, rows, expected):
    service.catalog_path.write_text("КодНоменклатуры|Номенклатура|Коллекция\n" + rows, encoding="utf-8")
    result = service.load("C")
    assert result.products[0].collection == expected
    assert all(not p.collection for p in result.products[1:])
    assert tuple(p.recommendation for p in result.products) == service.inference.get_recommendations.return_value.recommendations


def test_collection_catalog_is_cached_and_refreshed_without_per_card_reads(service, monkeypatch):
    from pathlib import Path
    path = service.catalog_path
    path.write_text("КодНоменклатуры|Номенклатура|Коллекция\nORIGINAL-1|Товар|Весна\n", encoding="utf-8")
    original = Path.open
    reads = []
    def open_path(self, *args, **kwargs):
        if self == path:
            reads.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_path)
    assert service.load("C").products[0].collection == "Весна"
    assert service.load("C").products[0].collection == "Весна"
    assert len(reads) == 1
    path.write_text("КодНоменклатуры|Номенклатура|Коллекция\nORIGINAL-1|Товар|Осень-Зима\n", encoding="utf-8")
    assert service.load("C").products[0].collection == "Осень-Зима"
