"""Чистые адаптеры отдельных raw объектов. Экспорт на диск не выполняется."""

from ._common import AdapterError
from .actions import adapt_action
from .orders import adapt_order
from .customers import adapt_customer
from .customer_merges import adapt_customer_merge
