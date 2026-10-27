"""Synthetic demonstration only; never loads customer history or model artifacts."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Application.evaluation.experiments.customer_interest import (
    Background, InterestAction, InterestConfig, InterestLine, InterestOrder, InterestSnapshot,
    ResearchContext, build_profile, score_item,
)
from Application.interactions import InteractionType
from Application.mindbox.records import ProductKey
from Application.model.item_feature_snapshot import FEATURE_FIELDS
from Application.recommendation_catalog import item_from_metadata

CUTOFF = datetime(2026, 7, 1, tzinfo=timezone.utc)
VALUES = {
    "ВидНоменклатуры": ("Пиджак", "Футболка"), "ВидАссортимента": ("Классический", "Повседневный"),
    "КатегорияНаСайте": ("Костюмы", "Трикотаж"), "Коллекция": ("Коллекция A", "Коллекция B"),
    "СтилеваяГруппа": ("Классика", "Кэжуал"), "Марка": ("Марка A", "Марка B"),
    "ПолНоменклатуры": ("Мужской", "Мужской"), "ГруппаСоставов": ("Шерсть", "Хлопок"),
    "СезонНоски": ("Осень", "Лето"),
}


def fixtures():
    items = [item_from_metadata(f"{prefix}{i:05d}", {name: VALUES[name][group] for name in FEATURE_FIELDS})
             for group, prefix in enumerate((1, 2)) for i in range(1, 7)]
    items += [item_from_metadata("300001", {"ВидНоменклатуры": "Пиджак"}), item_from_metadata("300002", {})]
    distributions = {name: ({values[0]: 1.} if values[0] == values[1] else {values[0]: .5, values[1]: .5})
                     for name, values in VALUES.items()}
    background = Background(distributions, 100, CUTOFF - timedelta(days=1), True)
    context = ResearchContext(CUTOFF, CUTOFF - timedelta(days=365), True, True)
    return items, background, context


def action(code, offset, kind=InteractionType.VIEW, *, identifier=None):
    when = CUTOFF - timedelta(days=offset)
    return InterestAction(identifier or f"{kind.value}-{code}-{offset}", "synthetic", when, kind,
                           (ProductKey("offline1C", code),), when + timedelta(minutes=1))


def order(code, offset, statuses=("CP",), *, quantity=Decimal(1), identifier=None):
    when = CUTOFF - timedelta(days=offset)
    snapshots = tuple(InterestSnapshot((InterestLine(ProductKey("offline1C", code), status, quantity, "stable-line"),),
                                       when + timedelta(minutes=n + 1)) for n, status in enumerate(statuses))
    return InterestOrder(identifier or f"order-{code}-{offset}", "synthetic", when, snapshots)


def demo():
    items, background, context = fixtures()
    scenarios = {
        "Нет истории": [],
        "Один просмотр": [action("100001", 1)],
        "Два просмотра разных товаров": [action("100001", 1), action("100002", 2)],
        "Несколько просмотров пиджаков": [action(f"10000{i}", i) for i in range(1, 4)],
        "Избранное — футболки": [action(f"20000{i}", i, InteractionType.FAVORITE) for i in range(1, 4)],
        "Покупки — пиджаки": [order(f"10000{i}", i, ("CP", "CP")) for i in range(1, 4)],
        "Отмены и возвраты без покупки": [order("100001", 1, ("otmenen",)), order("100002", 2, ("Return",))],
        "Конфликтующие статусы заказа": [order("100001", 1, ("CP", "otmenen"))],
        "Сбалансированные интересы": [action(f"{prefix}0000{i}", i, InteractionType.FAVORITE)
                                       for prefix in (1, 2) for i in range(1, 4)],
        "Смена интереса к пиджакам": [action(f"20000{i}", 200 + i, InteractionType.FAVORITE) for i in range(1, 4)]
                                      + [action(f"10000{i}", i, InteractionType.FAVORITE) for i in range(1, 4)],
    }
    results = []
    for title, events in scenarios.items():
        profile = build_profile("synthetic", events, items, background, context)
        matches = {label: asdict(score_item(profile, code)) for label, code in (
            ("Пиджак", "100006"), ("Футболка", "200006"), ("Пиджак только с видом", "300001"), ("Без атрибутов", "300002"))}
        results.append({"scenario": title, "profile_status": profile.status, "diagnostics": dict(profile.diagnostics),
                        "matches": matches})
    return {"dataset_kind": "SYNTHETIC", "as_of": CUTOFF.isoformat(), "mode": "STRICT",
            "purpose": "technical demonstration, no recommendation quality or probability claim",
            "configuration": asdict(InterestConfig()), "reference": "100 artificial users, balanced distributions; not real popularity",
            "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    protected = [ROOT / name for name in ("model", "input_data", "user_settings", ".git")]
    if output == ROOT or output in ROOT.parents or any(output == p or p in output.parents or output in p.parents for p in protected):
        parser.error("Use a separate research output directory")
    output.mkdir(parents=True, exist_ok=True)
    result = demo()
    with (output / "SYNTHETIC_EXAMPLES.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# Синтетические примеры — не рабочие рекомендации", "",
             "Значения — signed relative-interest utility, не вероятность покупки. Фон искусственный 50/50.", "",
             "| Сценарий | Статус профиля | Пиджак | Футболка |",
             "|---|---|---:|---:|"]
    for case in result["cases"]:
        matches = case["matches"]
        lines.append(f"| {case['scenario']} | {case['profile_status']} | {matches['Пиджак']['score']:.6f} | {matches['Футболка']['score']:.6f} |")
    lines += ["", "Подробные веса, фон, сглаженные доли, confidence и вклады каждого атрибута — SYNTHETIC_EXAMPLES.json.",
              "Снимки одного заказа не создают повторных покупок. Отмены/return-only/conflicts не создают положительного сигнала.",
              "При нехватке истории/данных итоговый score=0: сохранить исходный ranking, не придумывать персонализацию."]
    (output / "SYNTHETIC_EXAMPLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"synthetic_cases": len(result["cases"]), "ranking_changed": False, "training_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
