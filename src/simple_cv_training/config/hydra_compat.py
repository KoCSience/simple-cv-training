from __future__ import annotations

import argparse
from typing import Any


def patch_hydra_argparse_for_python314() -> None:
    """Make Hydra 1.3 argparse help compatible with Python 3.14.

    Python 3.14 checks whether "%" is contained in argparse help strings.
    Hydra 1.3 uses a lazy help object for shell completion, so adding
    containment support keeps Hydra's parser construction working.
    """

    if getattr(argparse.ArgumentParser.add_argument, "_scv_hydra_compat", False):
        return

    original_add_argument = argparse.ArgumentParser.add_argument

    def add_argument_with_string_help(
        self: argparse.ArgumentParser,
        *args: Any,
        **kwargs: Any,
    ) -> argparse.Action:
        help_value = kwargs.get("help")
        if help_value is not None and not isinstance(help_value, str):
            kwargs["help"] = str(help_value)
        return original_add_argument(self, *args, **kwargs)

    add_argument_with_string_help._scv_hydra_compat = True  # type: ignore[attr-defined]
    argparse.ArgumentParser.add_argument = add_argument_with_string_help
