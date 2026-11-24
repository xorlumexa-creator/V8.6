"""Install self-check (pure Python, no CAD kernel, no network).

Catches the packaging bug where main.py imports a sibling module that the Docker image never copies - the
endpoints then fail silently with 503 ("modules did not load").
Run:  python tests/test_install_modules.py   (or python -m pytest tests/test_install_modules.py)
"""
import ast
import glob
import importlib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _local_modules():
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, "*.py"))}


def _local_imports(path, local):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return (found & local) - {os.path.splitext(os.path.basename(path))[0]}


def needed_modules():
    """Every sibling module main.py needs, including the ones those modules import in turn."""
    local = _local_modules()
    todo, seen = ["main"], set()
    while todo:
        name = todo.pop()
        for dep in _local_imports(os.path.join(ROOT, name + ".py"), local):
            if dep not in seen:
                seen.add(dep)
                todo.append(dep)
    return seen


def _dockerfile():
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as fh:
        return fh.read()


def copied_modules():
    """Module names the Dockerfile's COPY lines bring into /app; None means 'all top-level .py files'."""
    names = set()
    for line in _dockerfile().splitlines():
        parts = line.strip().split()
        if not parts or parts[0].upper() != "COPY":
            continue
        args = [p for p in parts[1:] if not p.startswith("--")]
        for src in args[:-1]:                       # the last argument is the destination
            if src in (".", "./", "*.py"):
                return None
            if src.endswith(".py"):
                names.add(os.path.splitext(os.path.basename(src))[0])
    return names


def test_every_module_main_needs_is_copied_into_the_image():
    copied = copied_modules()
    if copied is None:
        return
    missing = sorted(needed_modules() - copied)
    assert not missing, f"Dockerfile does not copy: {missing}"


def test_dockerfile_build_guard_imports_the_design_modules():
    guard = " ".join(l for l in _dockerfile().splitlines() if l.strip().upper().startswith("RUN") and "python -c" in l)
    missing = sorted(m for m in needed_modules() if not re.search(rf"\b{m}\b", guard))
    assert not missing, f"the build-time import guard does not cover: {missing}"


def test_design_modules_import():
    for name in sorted(needed_modules()):
        importlib.import_module(name)


if __name__ == "__main__":
    failed = 0
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", fn_name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", fn_name, "-", exc)
    sys.exit(1 if failed else 0)
