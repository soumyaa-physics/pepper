#!/usr/bin/env python3
"""
json_to_toml.py — Convert between JSON and TOML files.

Direction is auto-detected from the input file's extension (.json or .toml).

Usage:
    python json_to_toml.py config.hjson          # -> config.toml
    python json_to_toml.py config.toml           # -> config.json
    python json_to_toml.py config.hjson out.toml  # custom output path

Dependencies:
    pip install hjson tomli-w
"""

import sys
import hjson
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

try:
    import tomli
except ImportError:
    logger.error(
        "The 'tomli' module is not installed. "
        "Please install it to run this script."
        )
    raise

try:
    import tomli_w
except ImportError:
    logger.error(
        "The 'tomli_w' module is not installed. "
        "Please install it to run this script."
        )
    raise


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python hjson_toml.py <input.json|input.toml> [output_path]")

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        sys.exit(f"Error: file not found: {input_path}")

    ext = input_path.suffix.lower()
    if ext == ".json":
        data = hjson.loads(input_path.read_text(encoding="utf-8"))
        output_text = tomli_w.dumps(data)
        default_output = input_path.with_suffix(".toml")
    elif ext == ".toml":
        data = tomli.loads(input_path.read_text(encoding="utf-8"))
        output_text = hjson.dumpsJSON(data, indent=4)
        default_output = input_path.with_suffix(".json")
    else:
        sys.exit(f"Error: unrecognized extension '{ext}' — expected .json or .toml")

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output
    output_path.write_text(output_text, encoding="utf-8")
    print(f"Converted '{input_path}' -> '{output_path}'")


if __name__ == "__main__":
    main()
