"""Small detail-job boundary: metadata preflight, paths and strict safe IPC."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import os
import re
from pathlib import Path

from Application.paths import PROJECT_ROOT
from Application.statistics_detail_contract import DetailPeriod, OrderMode
from Application.statistics_detail_storage import DetailExportError, outside_repository
from Application.statistics_storage_safety import reject_reparse
from Application.mindbox.canonical_storage import catalog, checked_directory, effective_entries
from Application.mindbox.raw_reader import part_files

MAX_MESSAGE = 4096
STAGES = {
    'source_validation': 'Проверка источников',
    'history': 'Подготовка истории заказов',
    'personal_metrics': 'Расчёт показателей покупателей',
    'rfm': 'Расчёт персонального RFM',
    'clients': 'Запись клиентов',
    'orders': 'Запись заказов',
    'order_lines': 'Запись товарных позиций',
    'zip_verification': 'Проверка архива',
    'atomic_save': 'Сохранение ZIP',
}
ERRORS = {
    'INVALID_OPTIONS': 'Проверьте даты и параметры выгрузки.',
    'SOURCE_NOT_READY': 'Нет готовых canonical Orders и CustomerMerges. Проверьте сохранённые источники.',
    'INVALID_EXPORT_PATH': 'Выберите ZIP и временный каталог вне проекта, источников и моделей.',
    'SYNCHRONIZED_DESTINATION': 'Выберите локальный каталог вне облачной синхронизации.',
    'PRIVATE_TEMPORARY_ROOT_REQUIRED': 'Выберите отдельный приватный временный каталог.',
    'TEMPORARY_ROOT_MUST_BE_ON_D': 'Временный каталог должен находиться на диске D:.',
    'PRIVATE_OBJECT_ACL_FAILED': 'Не удалось защитить временные файлы. Проверьте права каталога.',
    'ACL_INSPECTION_FAILED': 'Не удалось проверить права. Проверьте среду Windows PowerShell и доступ.',
    'UNSAFE_STORAGE_ACL': 'Права каталога не обеспечивают безопасное размещение выгрузки.',
    'PRIVATE_INDEX_ROOT_REQUIRED': 'Временный каталог должен иметь приватные права текущего пользователя.',
    'INSUFFICIENT_SPACE': 'Недостаточно свободного места. Освободите место или выберите другой каталог.',
    'TEMPORARY_BUDGET_EXCEEDED': 'Превышен лимит временного состояния.',
    'ACCESS_DENIED': 'Отказано в доступе. Проверьте права и не открыт ли файл назначения.',
    'IO_ERROR': 'Ошибка чтения или сохранения. Проверьте доступность источников и ZIP.',
    'DESTINATION_CHANGED': 'Файл назначения изменился во время экспорта. Повторите выбор пути.',
    'SOURCE_CHANGED': 'Источники изменились. Повторите экспорт после завершения обновления.',
    'DETAIL_EXPORT_FAILED': 'Backend не завершил выгрузку. Проверьте источники, доступ и свободное место.',
    'SENSITIVE_STATE_CLEANUP_FAILED': 'Не удалось безопасно удалить временную БД. Требуется проверка указанного пути.',
    'SENSITIVE_STATE_CLOSE_FAILED': 'Не удалось закрыть временную БД. Требуется проверка указанного пути.',
    'SENSITIVE_ARCHIVE_CLEANUP_FAILED': 'Не удалось безопасно удалить временный ZIP. Требуется проверка указанного пути.',
    'CONTROL_PROTOCOL_ERROR': 'Нарушен канал управления worker.',
    'WORKER_PROTOCOL_ERROR': 'Некорректное служебное сообщение worker. Запрошена безопасная отмена.',
    'WORKER_CRASHED': 'Worker завершился аварийно. Могли остаться временные файлы.',
    'FAILED_TO_START': 'Не удалось запустить Python-процесс экспорта.',
    'FORCED_STOP': 'Процесс завершён принудительно. Очистка не гарантируется; проверьте временные файлы.',
}
ERRORS.update({
    'ORDER_SNAPSHOT_CONFLICT': 'Обнаружены конфликтующие снимки заказов. Проверьте canonical-источник.',
    'SOURCE_DATA_INVALID': 'Некорректные исходные записи. Экспорт остановлен без публикации ZIP.',
    'SOURCE_PARTS_MISMATCH': 'Набор частей canonical не соответствует каталогу.',
    'CUSTOMER_MERGES_REQUIRED': 'Для выгрузки нужны сохранённые CustomerMerges.',
    'INVALID_CUSTOMERS_SCHEMA': 'Не поддерживается схема canonical Customers.',
    'CUSTOMERS_CHANGED': 'Профили Customers изменились во время выгрузки.',
    'INCONSISTENT_PROFILE_IDENTITY': 'Профили Customers не согласованы с идентификаторами источника.',
    'LOCAL_FIXED_VOLUME_REQUIRED': 'Выберите локальный фиксированный диск; сетевой путь не поддерживается.',
    'WRITABLE_NTFS_REQUIRED': 'Для выгрузки нужен доступный для записи том NTFS.',
    'UNTRUSTED_STORAGE_OWNER_OR_ACL': 'Владелец или права каталога не прошли проверку.',
    'OWNER_FULL_CONTROL_REQUIRED': 'Текущему пользователю нужны полные права на приватный каталог.',
    'INVALID_EXPORT_OPTIONS': 'Некорректные ограничения или параметры экспорта.',
})
for _code in ('CSV_SCHEMA_MISMATCH', 'ROW_COUNT_MISMATCH', 'BROKEN_DETAIL_RELATIONS', 'ARCHIVE_MEMBERS_MISMATCH',
              'ALGORITHM_CHANGED', 'PUBLICATION_PATH_CHANGED', 'PROGRESS_CALLBACK_FAILED', 'CANCELLATION_CALLBACK_FAILED'):
    ERRORS[_code] = 'Не пройдена внутренняя проверка выгрузки. ZIP не опубликован; проверьте служебный код.'
CLEANUP_CODES = frozenset(k for k in ERRORS if k.startswith('SENSITIVE_'))


@dataclass(frozen=True, slots=True)
class SourceInfo:
    first_date: date
    last_date: date


def inspect_sources(root, *, today=None):
    """Only catalog, lock metadata and part filenames; never Orders/Customers rows."""
    try:
        root = reject_reparse(root)
        lock = root/'canonical/.writer.lock'
        if not (root/'canonical/catalog.json').is_file() or not lock.is_file() or lock.stat().st_size < 1:
            raise ValueError
        data = catalog(root)
        entries = effective_entries(data, 'orders')
        merge = data['customer_merges']
        if not entries or not merge:
            raise ValueError
        for name, entry in [('orders', e) for e in entries]+[('customer_merges', merge)]:
            directory = reject_reparse(checked_directory(root, entry['directory'], name))
            if not directory.is_dir() or len(part_files(directory, name)) != entry['parts']:
                raise ValueError
        current = today or datetime.now(timezone.utc).date()
        ranges = [(datetime.fromisoformat(e['since']), datetime.fromisoformat(e['until'])) for e in entries]
        ranges = [(a, b) for a, b in ranges if a.tzinfo and b.tzinfo and a < b and a.date() <= current]
        if not ranges:
            raise ValueError
        return SourceInfo(min(a.date() for a, _ in ranges),
                          min(current, max((b-timedelta(microseconds=1)).date() for _, b in ranges)))
    except Exception:
        raise DetailExportError('SOURCE_NOT_READY') from None


def default_temporary_root():
    return Path(os.environ.get('KANZLER_DETAIL_TEMP_ROOT') or
                PROJECT_ROOT.parent/(PROJECT_ROOT.name+'-Private')/'detail-export-temp')


def validate_period(as_of, period, mode):
    if (type(as_of) is not date or not isinstance(period, DetailPeriod) or not isinstance(mode, OrderMode)
            or as_of == date.max or period.end_date is not None and period.end_date > as_of):
        raise DetailExportError('INVALID_OPTIONS')


def validate_paths(raw_root, destination, temporary_root):
    """Fast lexical/namespace checks only. Real ACL/volume checks stay in worker."""
    root, target, temporary = map(reject_reparse, (raw_root, destination, temporary_root))
    forbidden = {'.git', '.venv312', 'canonical', 'statistics_index', 'model', 'models', 'input_data'}
    for path in (target, temporary):
        if (not outside_repository(path if path.is_dir() else path.parent)
                or path.is_relative_to(PROJECT_ROOT) or path.is_relative_to(root)
                or any(p.lower() in forbidden for p in path.parts)):
            raise DetailExportError('INVALID_EXPORT_PATH')
        sync_roots = [Path(os.path.abspath(v)) for k in ('OneDrive', 'OneDriveCommercial', 'OneDriveConsumer')
                      if (v := os.environ.get(k))]
        if (any(path.is_relative_to(p) for p in sync_roots)
                or any(p.lower().startswith(('onedrive', 'dropbox', 'google drive')) for p in path.parts)):
            raise DetailExportError('SYNCHRONIZED_DESTINATION')
    if (target.suffix.lower() != '.zip' or not target.parent.is_dir() or target.is_dir()
            or target.is_relative_to(temporary)):
        raise DetailExportError('INVALID_EXPORT_PATH')
    if os.name == 'nt' and temporary.drive.upper() != 'D:':
        raise DetailExportError('TEMPORARY_ROOT_MUST_BE_ON_D')
    return root, target, temporary


def validate_message(message):
    """Never return arbitrary strings from worker to the GUI."""
    if not isinstance(message, dict):
        raise ValueError('Invalid detail IPC')
    event = message.get('event')
    keys = set(message)
    if event == 'progress' and keys == {'event', 'stage', 'processed'}:
        if message['stage'] in STAGES and type(message['processed']) is int and 0 <= message['processed'] < 2**63:
            return message
    elif event == 'finished' and keys == {'event', 'rows'}:
        rows = message['rows']
        if (isinstance(rows, dict) and set(rows) == {'clients', 'orders', 'order_lines'}
                and all(type(n) is int and 0 <= n < 2**63 for n in rows.values())):
            return message
    elif event == 'cancelled' and keys == {'event'}:
        return message
    elif event == 'error' and keys in ({'event', 'code'}, {'event', 'code', 'technical_path'}):
        code = message['code']
        if isinstance(code, str) and code in ERRORS:
            path = message.get('technical_path')
            if path is None and keys == {'event', 'code'}:
                return message
            if code in CLEANUP_CODES and isinstance(path, str) and 0 < len(path) <= 1024 and not any(c in path for c in '\r\n\x00'):
                return message
    raise ValueError('Invalid detail IPC')


def encode_message(message):
    validate_message(message)
    data = json.dumps(message, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()+b'\n'
    if len(data) > MAX_MESSAGE:
        raise ValueError('Oversized detail IPC')
    return data


def decode_message(line):
    if not line or len(line) > MAX_MESSAGE:
        raise ValueError('Invalid detail IPC')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate detail IPC key')
            result[key] = value
        return result
    return validate_message(json.loads(line, object_pairs_hook=pairs))


def allowed_cleanup_path(value, temporary, destination):
    candidate = Path(value)
    return candidate.is_absolute() and (
        candidate.parent == Path(temporary) and re.fullmatch(r'detail-[a-z0-9_]{8}', candidate.name) is not None
        or candidate.parent == Path(destination).parent
        and re.fullmatch(r'\.statistics-detail-[0-9a-f]{32}\.partial', candidate.name) is not None)
