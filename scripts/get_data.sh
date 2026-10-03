#!/usr/bin/env bash
# Download the TumSeg database (2.2 GB, CC-BY) and convert it to 0.42 mm npz.
# Jensen et al., "3D whole body preclinical micro-CT database of subcutaneous
# tumors in mice with annotations from 3 annotators", Sci Data 2024.
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST/raw"
curl -L -C - -o "$DEST/raw/tumseg.zip" \
  "https://erda.ku.dk/archives/ba4fcd9bfa0fb581d593297dd43d1fd1/TumSeg%20database.zip"
unzip -q -o "$DEST/raw/tumseg.zip" -d "$DEST/raw"
python3 scripts/preprocess.py --src "$DEST/raw/TumSeg database" --out "$DEST/npz"
echo "export FAUXGRAFT_DATA=$DEST/npz"
