"""Bounded manual-import context. Diagnostics contain schema names and counters only."""
from dataclasses import asdict, dataclass
from contextlib import closing
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from .adapters._common import AdapterError
from .customers_stream import iter_json_records
from .manual_sources import ManualImportError, normalize_sources, natural_path_key
from .raw_reader import EXPORT_ROOTS, RawExportError

TITLES = {'actions': 'Действия', 'orders': 'Заказы', 'customers': 'Клиенты'}
MESSAGES = {
    'missing': 'отсутствует обязательное поле', 'object': 'ожидается объект',
    'objects': 'ожидается массив объектов', 'string': 'ожидается строка',
    'number': 'ожидается число', 'finite': 'число должно быть конечным',
    'integer': 'ожидается целое число', 'boolean': 'ожидается логическое значение',
    'identifier': 'идентификатор должен быть строкой или целым числом',
    'empty_identifier': 'пустой идентификатор или краевые пробелы',
    'date': 'некорректная дата или время', 'namespace': 'отсутствует поддерживаемый идентификатор товара или выбрано несколько источников',
    'root_object': 'корневое значение JSON должно быть объектом',
    'root_key': 'неверный корневой ключ экспорта', 'root_array': 'корневое поле должно содержать массив',
    'record_object': 'элементы корневого массива должны быть объектами',
    'duplicate_key': 'JSON содержит повторяющиеся ключи', 'incomplete_json': 'неожиданное завершение JSON-файла',
    'trailing_json': 'лишнее содержимое после корневого объекта JSON',
    'json_structure': 'некорректная структура JSON', 'encoding': 'неверная кодировка UTF-8',
    'file_missing': 'выбранный файл больше недоступен', 'file_access': 'нет доступа к файлу',
    'file_read': 'ошибка чтения или записи файла', 'unknown': 'внутренняя ошибка обработки; исходные данные не публикуются',
    'incomplete_export': 'обнаружена незавершённая часть экспорта',
}
ADAPTER_REASONS = {
    'отсутствует обязательное поле': 'missing', 'ожидается вложенный object': 'object',
    'ожидается object': 'object', 'ожидается массив объектов': 'objects', 'ожидается строка': 'string',
    'ожидается JSON number': 'number', 'некорректное число': 'number', 'число должно быть конечным': 'finite',
    'ожидается integer': 'integer', 'ожидается boolean': 'boolean',
    'identifier должен быть string или integer': 'identifier', 'пустой identifier или краевые пробелы': 'empty_identifier',
    'некорректный timestamp': 'date', 'некорректное смещение UTC': 'date',
    'некорректная дата или время': 'date', 'некорректная дата': 'date',
    'отсутствует поддержанный namespace или неоднозначность нескольких namespaces': 'namespace',
}
FIELDS = frozenset({
    'ids.mindboxId', 'ids.retailOrderId', 'customer.ids.mindboxId', 'firstAction.dateTimeUtc',
    'firstAction.channel.ids.externalId', 'firstAction.channel.name', 'totalPrice', 'deliveryCost',
    'lines', 'number', 'id', 'product', 'product.ids', 'product.name', 'quantity', 'basePricePerItem',
    'priceOfLine', 'status.ids.externalId', 'ids', 'ids.offline1C', 'ids.kanzlerKz',
    'product.ids.offline1C', 'product.ids.kanzlerKz', 'actionTemplate.ids.systemName', 'actionTemplate.name',
    'dateTimeUtc', 'creationDateTimeUtc', 'brand.ids.systemName', 'channel.ids.systemName',
    'channel.ids.externalId', 'channel.ids.mindboxId', 'channel.name', 'products', 'productCategories',
    'productView.price', 'productView.isAvailable', 'changeDateTimeUtc', 'birthDate', 'sex', 'firstName',
    'lastName', 'middleName', 'email', 'mobilePhone', 'ianaTimeZone', 'timeZoneSource', 'customFields',
    'lastActivatedCard', 'segmentations', 'balances', 'discountCards', 'subscriptions', 'orders', 'customerActions', 'customers',
})


def _field(path):
    if not isinstance(path, str):
        return False
    if path in FIELDS:
        return True
    import re
    match = re.fullmatch(r'lines\[(0|[1-9][0-9]{0,8})\]\.(.+)', path)
    return match is not None and match[2] in FIELDS


@dataclass(frozen=True)
class ManualImportDiagnostic:
    source: str
    phase: str
    file_index: int
    file_total: int
    record_index: int | None = None
    line_index: int | None = None
    field_path: str | None = None
    error_category: str = 'unknown'

    def __post_init__(self):
        if self.source not in TITLES or self.phase not in {'selection', 'copy', 'read', 'validation'} or self.error_category not in MESSAGES:
            raise ValueError('Invalid manual diagnostic category')
        if type(self.file_index) is not int or type(self.file_total) is not int or not 1 <= self.file_index <= self.file_total <= 10**9:
            raise ValueError('Invalid manual diagnostic file counters')
        for value in (self.record_index, self.line_index):
            if value is not None and (type(value) is not int or not 0 <= value <= 10**18):
                raise ValueError('Invalid manual diagnostic record counters')
        if self.line_index is not None and (self.source != 'orders' or self.line_index < 1):
            raise ValueError('Invalid manual diagnostic line counter')
        if self.field_path is not None and not _field(self.field_path):
            raise ValueError('Invalid manual diagnostic field path')

    def to_dict(self):
        return {**asdict(self), 'safe_message': MESSAGES[self.error_category]}

    def format(self):
        phase = {'selection': 'выбора', 'copy': 'копирования', 'read': 'чтения JSON', 'validation': 'проверки'}[self.phase]
        text = f'{TITLES[self.source]} → ошибка {phase} файла {self.file_index} из {self.file_total}'
        if self.record_index is not None:
            label = 'после записи' if self.phase == 'read' else 'запись'
            text += f', {label} ' + format(self.record_index, ',').replace(',', ' ')
        text += '.'
        if self.line_index is not None:
            text += '\nПозиция заказа: ' + format(self.line_index, ',').replace(',', ' ') + '.'
        if self.field_path is not None:
            text += '\nПоле: ' + self.field_path + '.'
        return text + '\nПричина: ' + MESSAGES[self.error_category] + '.'


def format_manual_diagnostic(value):
    try:
        if not isinstance(value, dict):
            raise ValueError
        # Ignore even a supplied safe_message; reconstruct from the closed categories.
        return ManualImportDiagnostic(**{key: value[key] for key in ManualImportDiagnostic.__dataclass_fields__}).format()
    except (KeyError, TypeError, ValueError):
        return 'Ошибка ручного импорта → некорректное описание причины от процесса.'


class ManualImportContext:
    def __init__(self, source, file_total):
        self.source, self.file_total = source, file_total
        self.file_index, self.record_index = 1, 0

    def fail(self, exc, phase='validation'):
        field, line, category = None, None, 'unknown'
        if isinstance(exc, AdapterError):
            message = exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) else ''
            path, separator, reason = message.partition(': ')
            category = ADAPTER_REASONS.get(reason, 'unknown') if separator else 'unknown'
            field = path if category != 'unknown' and path in FIELDS else None
            if self.source == 'orders' and type(exc.line_index) is int and 1 <= exc.line_index <= 10**9:
                line = exc.line_index
                if field:
                    if field == 'ids' or field in {'ids.offline1C', 'ids.kanzlerKz'}:
                        field = 'product.' + field
                    field = f'lines[{line - 1}].{field}'
        elif isinstance(exc, RawExportError):
            category = exc.category if exc.category in MESSAGES else 'json_structure'
            if category in {'root_key', 'root_array'}:
                field = EXPORT_ROOTS[self.source]
        elif isinstance(exc, OSError):
            category = ('file_missing' if isinstance(exc, FileNotFoundError) else
                        'file_access' if isinstance(exc, PermissionError) else 'file_read')
        diagnostic = ManualImportDiagnostic(self.source, phase, self.file_index, self.file_total,
            None if phase in {'selection', 'copy'} or (phase == 'validation' and not self.record_index) else self.record_index,
            line, field, category)
        raise ManualImportError(diagnostic.format(), diagnostic=diagnostic) from None

    def records(self, paths, *, start_index=1):
        for index, path in enumerate(paths, start_index):
            self.file_index, self.record_index = index, 0
            try:
                if path.name.endswith(('.tmp', '.download')):
                    raise RawExportError('Incomplete export part', category='incomplete_export')
                with closing(iter_json_records(path, EXPORT_ROOTS[self.source])) as records:
                    for raw in records:
                        self.record_index += 1
                        yield raw
            except InterruptedError:
                raise
            except Exception as exc:
                self.fail(exc, 'read')


def normalize_manual_sources(sources, source):
    """Keep normalization semantics; add an ordinal when a selected file is missing."""
    try:
        return normalize_sources(sources)
    except ManualImportError as original:
        values = (sources,) if isinstance(sources, (str, PathLike)) else sources
        if not isinstance(values, Sequence) or not values or not all(isinstance(value, (str, PathLike)) for value in values):
            raise
        try:
            paths = sorted((Path(value).resolve() for value in values), key=natural_path_key)
        except (OSError, RuntimeError, TypeError):
            raise original from None
        context = ManualImportContext(source, len(paths))
        for index, path in enumerate(paths, 1):
            context.file_index = index
            try:
                available = path.is_file()
            except OSError as exc:
                context.fail(exc, 'selection')
            if not available:
                context.fail(FileNotFoundError(), 'selection')
        raise
