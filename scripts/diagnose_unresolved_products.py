"""Local preflight-only product audit. No trainer, publication, API or test scoring."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit_preflight(manifest, raw_root, catalog, cfg):
    from Application.model import BPRMF as core, mindbox_production_training as production
    from Application.model.product_resolution_audit import UnresolvedProductAudit
    from Application.product_resolution import ProductResolver, load_catalog
    audit = UnresolvedProductAudit(load_catalog(catalog))
    original = ProductResolver.resolve_interaction
    def observe(resolver, interaction, *, strict=True):
        resolved = original(resolver, interaction, strict=strict)
        audit.observe(resolver, interaction, resolved)
        return resolved
    def forbidden(*args, **kwargs):
        raise AssertionError('Diagnostic scan must never train or publish')
    with patch.object(ProductResolver, 'resolve_interaction', observe), \
            patch.object(core, 'train_prepared_data_with_metrics', forbidden), \
            patch.object(production, 'train_and_publish_production_model', forbidden):
        result = production.preflight_production_training(
            manifest, raw_root=raw_root, catalog_path=catalog, cfg=cfg)
    report = audit.report()
    report['preflight'] = {
        'error_code': result.error_code, 'dataset': dict(result.dataset),
        'quality': None if result.quality_report is None else {
            'level': result.quality_report.level.value,
            'training_allowed': result.quality_report.training_allowed,
            'metrics': dict(result.quality_report.metrics),
            'issues': [{'code': issue.code, 'level': issue.level.value, 'count': issue.count,
                        'rate': issue.rate, 'message': issue.message} for issue in result.quality_report.issues]},
        'training_started': result.training_started, 'publication_started': result.publication_started,
        'published': result.published,
    }
    if result.quality_report is None:
        raise ValueError('Diagnostic preparation failed; no partial audit presented as complete')
    assert report['total_events'] == result.dataset['unresolved_products']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'input_data/MindboxRaw/canonical/training.json')
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'input_data/MindboxRaw')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'input_data/nomenclature.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from Application.tabs.training_workflow import production_config
    from Application.model.BPRMF import _load_train_config_from_json
    values = {'epochs': 50, 'w_purchase': 10., 'w_favorite': 2., 'w_view_item': .5}
    config = production_config(values, ROOT / 'user_settings', args.catalog.resolve().parent)
    with tempfile.TemporaryDirectory(prefix='quality-preflight-') as temporary:
        path = Path(temporary) / 'config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        cfg = _load_train_config_from_json(str(path))
    report = audit_preflight(args.manifest, args.raw_root, args.catalog, cfg)
    report['preparation_config'] = {key: value for key, value in asdict(cfg).items()
                                    if key in ('w_purchase', 'w_favorite', 'w_view_item', 'min_user_interactions_for_eval')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('products', 'preflight')}, ensure_ascii=False))
    print('Preflight:', report['preflight']['quality']['level'], 'training_started=False; publication_started=False')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
