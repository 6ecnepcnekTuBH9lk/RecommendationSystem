"""Ошибки конфигурации, транспорта и публикации экспортов Mindbox."""


class MindboxError(Exception):
    """Базовая ошибка изолированного слоя Mindbox."""


class MindboxConfigError(MindboxError):
    """Отсутствующая или некорректная локальная конфигурация."""


class MindboxApiError(MindboxError):
    """HTTP-ошибка, ошибка операции или нарушение контракта ответа."""


class MindboxExportCancelledError(MindboxApiError):
    """Mindbox отменил экспорт."""


class MindboxExportTimeoutError(MindboxApiError):
    """Истекло время ожидания готовности экспорта."""


class MindboxDownloadError(MindboxError):
    """Не удалось скачать или опубликовать все части выгрузки."""
