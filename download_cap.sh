#!/bin/bash
BASE="https://physionet.org/files/capslpdb/1.0.0"
mkdir -p cap_data

for i in $(seq 1 16); do
    SUB="n${i}"
    for EXT in edf txt; do
        DEST="cap_data/${SUB}.${EXT}"
        if [ -f "$DEST" ] && [ $(stat -c%s "$DEST") -gt 1000 ]; then
            echo "[OK] ${SUB}.${EXT} already exists"
        else
            echo "[GET] ${SUB}.${EXT}..."
            wget -q --show-progress \
                -O "$DEST" \
                "${BASE}/${SUB}.${EXT}"
            echo "[DONE] $(ls -lh $DEST | awk '{print $5}')"
        fi
    done
done
echo "All 16 healthy subjects downloaded"
