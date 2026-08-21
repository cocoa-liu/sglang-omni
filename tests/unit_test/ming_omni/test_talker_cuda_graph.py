# SPDX-License-Identifier: Apache-2.0
"""Ming talker device graph source-level regression tests."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TALKER_SOURCE = (
    _REPO_ROOT
    / "sglang_omni"
    / "models"
    / "ming_omni"
    / "talker"
    / "modeling_ming_omni_talker.py"
)


def _method_node(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_ming_lazy_graph_captures_use_device_runtime() -> None:
    tree = ast.parse(_TALKER_SOURCE.read_text())
    methods = [
        ("CFMGraphExecutor", "_initialize_graph"),
        ("MingOmniTalker", "generate"),
    ]

    for class_name, method_name in methods:
        method = _method_node(tree, class_name, method_name)
        source = ast.unparse(method)
        assert ".new_graph()" in source
        assert ".graph_context(" in source
        assert "torch.cuda" not in source
