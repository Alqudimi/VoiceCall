#!/usr/bin/env bash
# Iraqi Voice Studio — premium pretrained bases (research: TITAN / Ov2Super
# beat the stock VCTK f0G40k/f0D40k for small-data fine-tuning).
# Usage:  bash scripts/colab/get_pretrained.sh [titan|ov2|stock] [32k|40k|48k]
# Installs the chosen G/D pair into assets/pretrained_v2/ under the standard
# names (f0G40k.pth / f0D40k.pth ...), backing up any previous files.
set -euo pipefail
WHICH="${1:-titan}"; SR="${2:-40k}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/assets/pretrained_v2"; mkdir -p "$DEST"

case "$WHICH" in
  titan)
    BASE="https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/${SR}/pretrained"
    G="G-f0${SR}-TITAN-Medium.pth"; D="D-f0${SR}-TITAN-Medium.pth" ;;
  ov2)
    BASE="https://huggingface.co/ORVC/Ov2Super/resolve/main"
    G="f0G${SR}.pth"; D="f0D${SR}.pth" ;;
  stock)
    echo "keeping the official stock pretrained base"; exit 0 ;;
  *) echo "unknown pretrain: $WHICH (titan|ov2|stock)"; exit 2 ;;
esac

TG="$DEST/f0G${SR}.pth"; TD="$DEST/f0D${SR}.pth"
for f in "$TG" "$TD"; do [ -f "$f" ] && cp "$f" "$f.stock.bak" && echo "backed up $f"; done
echo "downloading $WHICH $SR from $BASE ..."
curl -fL --retry 3 -o "$TG" "$BASE/$G"
curl -fL --retry 3 -o "$TD" "$BASE/$D"
ls -lh "$TG" "$TD"
echo "done — training will now fine-tune from $WHICH."
