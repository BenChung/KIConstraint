"""Build script to package the KiConstraint plugin for KiCAD."""

import json
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PLUGINS_DIR = ROOT_DIR / "plugin" / "plugins"
DIST_DIR = ROOT_DIR / "dist"


def _get_plugin_name() -> str:
    plugin_json = PLUGINS_DIR / "plugin.json"
    with open(plugin_json) as f:
        metadata = json.load(f)
    return metadata["identifier"].rsplit(".", 1)[-1]


KICAD_VENV_SITE_PACKAGES = (
    r"C:\Users\benchung\AppData\Local\KiCad\10.0"
    r"\python-environments\com.github.benchung.kiconstraint\Lib\site-packages"
)


def build() -> Path:
    """Copy the plugins directory into dist as a ready-to-use plugin folder."""
    plugin_name = _get_plugin_name()
    dest = DIST_DIR / plugin_name

    if dest.exists():
        shutil.rmtree(dest)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        PLUGINS_DIR,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    _write_run_test(dest)
    return dest


def _write_run_test(plugin_dest: Path) -> None:
    """Write a standalone script that runs test_action using KiCAD's venv."""
    script = plugin_dest.parent / "run_test.py"
    script.write_text(
        f'''\
"""Run the test action using KiCAD's plugin virtual environment."""

import site
import sys
import os

site.addsitedir(r"{KICAD_VENV_SITE_PACKAGES}")

# Add the plugin directory so local imports work
plugin_dir = os.path.join(os.path.dirname(__file__), "{plugin_dest.name}")
sys.path.insert(0, plugin_dir)

from test_action import main

main()
'''
    )


def install(kicad_plugin_dir: Path | None = None) -> Path:
    """Install the plugin into the local KiCAD plugins directory."""
    if kicad_plugin_dir is None:
        kicad_plugin_dir = Path.home() / ".local" / "share" / "kicad" / "9.0" / "plugins"

    plugin_name = _get_plugin_name()
    dest = kicad_plugin_dir / plugin_name

    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(
        PLUGINS_DIR,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    _write_run_test(dest)
    return dest


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build KiConstraint KiCAD plugin")
    parser.add_argument(
        "command",
        choices=["build", "install"],
        help="'build' to create plugin directory in dist, 'install' to install locally",
    )
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        default=None,
        help="Override KiCAD plugins directory (for 'install' command)",
    )
    args = parser.parse_args()

    if args.command == "build":
        dest = build()
        print(f"Plugin built: {dest}")
    elif args.command == "install":
        dest = install(args.plugin_dir)
        print(f"Plugin installed to: {dest}")


if __name__ == "__main__":
    main()
