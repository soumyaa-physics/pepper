#!/bin/bash
set -euo pipefail


usage() {
    cat <<EOF
Usage: ./install.sh [-f|--force] [-h|--help]

  -f, --force   Overwrite environment.sh with a fresh copy of example/environment.sh
                (the old one is backed up to environment.sh.bak) and recreate .venv
  -h, --help    Show this help
EOF
}

FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        -f|--force) FORCE=1 ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "error: unknown option '$1'" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

# Check we're in the repo root by looking for setup.cfg with the expected content
if [ ! -f setup.cfg ] || ! grep -q "name = pepper" setup.cfg 2>/dev/null; then
    echo "error: install.sh must be run from the pepper repo root" >&2
    exit 1
fi


# Copy environment template to root only if it doesn't exist, so that users can customize it without affecting the repo.
# With --force, replace it with a fresh copy (keeping a backup of the old one).
if [ "$FORCE" -eq 1 ] && [ -f environment.sh ]; then
    cp environment.sh environment.sh.bak
    echo "Backed up existing environment.sh to environment.sh.bak"
    cp example/environment.sh environment.sh
    echo "Replaced environment.sh with a fresh copy of example/environment.sh"
elif [ ! -f environment.sh ]; then
     cp example/environment.sh environment.sh
else
    echo "Keeping existing environment.sh (use --force to replace it)"
fi



# Source it to get LCG (temporarily allow unset variables in LCG scripts)
set +u
source environment.sh
set -u

# Create venv with access to LCG system packages (--clear wipes an existing venv on --force)
VENV_ARGS=(--system-site-packages)
if [ "$FORCE" -eq 1 ]; then
    VENV_ARGS+=(--clear)
fi
python3 -m venv "${VENV_ARGS[@]}" .venv
source .venv/bin/activate

# Install package in editable mode
python3 -m pip install --upgrade --editable .

# Patch the placeholder path in environment.sh
VENV_PATH="$(realpath .venv)"
sed -i "s|# source /INSERT/ABSOLUTE/PATH/TO/YOUR/ENVIRONMENT/bin/activate|source ${VENV_PATH}/bin/activate|" environment.sh


echo "Done. Run 'source environment.sh' to activate."