from __future__ import annotations

import pytest

from simple_cv_core.registry import Registry, RegistryConflictError, RegistryLookupError


def test_registry_returns_registered_item() -> None:
    registry = Registry("example")
    registry.register("item", lambda: "value")

    builder = registry.get("item")

    assert builder() == "value"


def test_registry_rejects_empty_name() -> None:
    registry = Registry("example")

    with pytest.raises(ValueError):
        registry.register("", object())


def test_registry_can_treat_duplicate_as_error() -> None:
    registry = Registry("example")
    registry.register("item", object())

    with pytest.raises(RegistryConflictError):
        registry.register("item", object(), strict=True)


def test_registry_lists_available_names_on_lookup_error() -> None:
    registry = Registry("example")
    registry.register("known", object())

    with pytest.raises(RegistryLookupError, match="known"):
        registry.get("missing")
