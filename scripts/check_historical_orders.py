"""Read-only streaming Orders schema audit; no imports, merges, training or publication."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_sources(sources):
    from Application.mindbox.adapters import adapt_order
    from Application.mindbox.identity import CustomerIdResolver
    from Application.mindbox.manual_diagnostics import ManualImportContext
    from Application.mindbox.manual_sources import ManualImportError, normalize_sources
    from Application.mindbox.selection import DEFAULT_SELECTION
    from Application.interactions import InteractionBuilder

    sources = normalize_sources(sources)
    resolver = CustomerIdResolver()  # Schema audit only; no canonical CustomerMerges loaded.
    results = []
    for index, path in enumerate(sources, 1):
        before = path.stat()
        started = time.monotonic()
        context = ManualImportContext('orders', len(sources))
        counts, errors, samples = Counter(), Counter(), []
        builder = InteractionBuilder(DEFAULT_SELECTION.interaction_rules())
        complete, read_error = False, None
        records = context.records((path,), start_index=index)
        try:
            for raw in records:
                counts['orders'] += 1
                raw_lines = raw.get('lines')
                if isinstance(raw_lines, list):
                    counts['raw_lines'] += len(raw_lines)
                    missing = sum(isinstance(line, dict) and 'basePricePerItem' not in line for line in raw_lines)
                    null = sum(isinstance(line, dict) and line.get('basePricePerItem', False) is None for line in raw_lines)
                    counts['missing_base_orders'] += bool(missing)
                    counts['missing_base_lines'] += missing
                    counts['null_base_lines'] += null
                try:
                    lines = adapt_order(raw, resolver, product_namespaces=DEFAULT_SELECTION.order_product_namespaces)
                except Exception as exc:
                    try:
                        context.fail(exc)
                    except ManualImportError as error:
                        detail = error.diagnostic
                        field = detail.field_path
                        if field and field.startswith('lines['):
                            field = 'lines[].' + field.split('].', 1)[1]
                        errors[(field, detail.error_category)] += 1
                        if len(samples) < 5:
                            samples.append(detail.to_dict())
                else:
                    counts['valid_orders'] += 1
                    counts['adapted_lines'] += len(lines)
                    for line in lines:
                        if builder.from_order_line(line) is not None:
                            counts['purchase_interactions'] += 1
                            counts['purchases_without_base_price'] += line.base_price_per_item is None
                if counts['orders'] % 50000 == 0:
                    print(f"Файл {index} из {len(sources)}: проверено {counts['orders']:,} заказов".replace(',', ' '), flush=True)
            complete = True  # Includes strict validation of the entire JSON tail.
        except ManualImportError as error:
            read_error = error.diagnostic.to_dict()
        finally:
            records.close()
        after = path.stat()
        unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        results.append({'file_index': index, 'file_total': len(sources), 'bytes': before.st_size,
            'complete_json': complete, 'size_mtime_unchanged': unchanged, 'seconds': time.monotonic() - started,
            'counts': {key: counts[key] for key in ('orders', 'raw_lines', 'valid_orders', 'adapted_lines',
                'missing_base_orders', 'missing_base_lines', 'null_base_lines', 'purchase_interactions', 'purchases_without_base_price')},
            'schema_errors': [{'field_path': field, 'error_category': category, 'count': count}
                              for (field, category), count in errors.items()],
            'error_samples': samples, 'read_error': read_error})
        print(f"Файл {index}: {counts['orders']} заказов; JSON complete={complete}; schema errors={sum(errors.values())}", flush=True)
    return {'mode': 'read_only_schema_audit', 'canonical_merges_checked': False, 'import_performed': False, 'files': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if any(output == path.resolve() or (output.exists() and path.exists() and output.samefile(path))
               for path in args.files):
            raise ValueError('Report must not replace an input file')
        report = check_sources(args.files)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    except (KeyboardInterrupt, InterruptedError):
        return 130
    except Exception:
        print('Не удалось выполнить read-only проверку заказов. Проверьте доступность выбранных файлов.', flush=True)
        return 1
    return 0 if all(row['complete_json'] and row['size_mtime_unchanged'] and not row['schema_errors']
                    for row in report['files']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
