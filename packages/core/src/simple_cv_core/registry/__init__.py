from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar
import warnings

T = TypeVar("T")


class RegistryConflictError(ValueError):
    pass


class RegistryLookupError(KeyError):
    pass


class Registry:
    def __init__(self, label: str):
        self.label = label
        self._items: dict[str, object] = {}

    def register(self, name: str, item: T, *, strict: bool = False) -> T:
        normalized_name = normalize_registry_name(name)
        if normalized_name in self._items:
            message = f"{self.label} registry already contains {normalized_name!r}; keeping the newest registration."
            if strict:
                raise RegistryConflictError(message)
            warnings.warn(message, stacklevel=2)
        self._items[normalized_name] = item
        return item

    def get(self, name: str) -> object:
        normalized_name = normalize_registry_name(name)
        try:
            return self._items[normalized_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise RegistryLookupError(
                f"Unknown {self.label} {normalized_name!r}. Available {self.label}s: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


ModelBuilder = TypeVar("ModelBuilder", bound=Callable[..., object])
DatasetBuilder = TypeVar("DatasetBuilder", bound=Callable[..., object])

MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")


def normalize_registry_name(name: str) -> str:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("registry name must not be empty")
    return normalized_name


def register_model(name: str, *, strict: bool = False) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        return MODEL_REGISTRY.register(name, builder, strict=strict)

    return decorator


def get_model(name: str) -> object:
    return MODEL_REGISTRY.get(name)


def register_dataset(name: str, *, strict: bool = False) -> Callable[[DatasetBuilder], DatasetBuilder]:
    def decorator(builder: DatasetBuilder) -> DatasetBuilder:
        return DATASET_REGISTRY.register(name, builder, strict=strict)

    return decorator


def get_dataset(name: str) -> object:
    return DATASET_REGISTRY.get(name)
