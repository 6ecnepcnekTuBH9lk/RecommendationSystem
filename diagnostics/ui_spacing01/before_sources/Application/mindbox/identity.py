"""Направленные объединения customer ID; без зависимости от профилей клиентов."""

from collections.abc import Iterable

from .records import CustomerMergeRecord


class CustomerIdentityError(Exception):
    """Ошибка графа identity, без идентификаторов в диагностике."""


class CustomerIdResolver:
    def __init__(self, merges: Iterable[CustomerMergeRecord] = ()) -> None:
        self._parents: dict[str, str] = {}
        self._canonical: dict[str, str] = {}
        for merge in merges:
            if (not isinstance(merge, CustomerMergeRecord)
                    or not isinstance(merge.merged_customer_ids, tuple) or not merge.merged_customer_ids):
                raise CustomerIdentityError("Malformed merge: ожидается запись с непустым списком источников")
            target = self._valid_id(merge.resulting_customer_id)
            for source in merge.merged_customer_ids:
                source = self._valid_id(source)
                if source in self._parents and self._parents[source] != target:
                    raise CustomerIdentityError("Conflicting mappings: один source customer имеет разные targets")
                self._parents[source] = target
        # Проверяем ВСЕ компоненты до возврата готового resolver, независимо от порядка событий.
        for source in self._parents:
            self.resolve(source)

    @staticmethod
    def _valid_id(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise CustomerIdentityError("Customer identifier должен быть непустой строкой без краевых пробелов")
        return value

    @property
    def alias_count(self) -> int:
        return len(self._parents)

    def resolve(self, source_customer_id: str) -> str:
        current = self._valid_id(source_customer_id)
        trail = []
        seen = set()
        while current in self._parents and current not in self._canonical:
            if current in seen:
                raise CustomerIdentityError("Cycle: обнаружен цикл customer merges")
            seen.add(current)
            trail.append(current)
            current = self._parents[current]
        canonical = self._canonical.get(current, current)
        for alias in trail:
            self._canonical[alias] = canonical
        # Не кешируем неизвестные ID: память зависит только от merge aliases.
        return canonical
