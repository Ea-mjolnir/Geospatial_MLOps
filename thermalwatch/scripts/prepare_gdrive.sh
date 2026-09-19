#!/bin/bash
# ThermalWatch — Upload to Google Drive via rclone
# ==================================================
# Uploads everything directly to GDrive.
# No local tarballs needed — saves disk space.
#
# Run from: ~/Geospatial_MLOps/thermalwatch
# Requires: rclone configured with gdrive remote

set -e

GDRIVE="gdrive:thermalwatch"

echo "=== ThermalWatch GDrive Upload ==="
echo "Target: $GDRIVE"
echo ""

# Create GDrive structure
echo "[0/5] Creating GDrive folder structure..."
rclone mkdir "$GDRIVE/project"
rclone mkdir "$GDRIVE/data/wildfire/patches"
rclone mkdir "$GDRIVE/data/wildfire/indexes"
rclone mkdir "$GDRIVE/data/solar/patches"
rclone mkdir "$GDRIVE/data/solar/indexes"
rclone mkdir "$GDRIVE/data/indexes"
rclone mkdir "$GDRIVE/checkpoints/ssl"
rclone mkdir "$GDRIVE/checkpoints/finetune/wildfire"
rclone mkdir "$GDRIVE/checkpoints/finetune/solar"
echo "  ✅ Folders created"

# 1. Bundle + upload code
echo ""
echo "[1/5] Bundling + uploading code..."
tar -czf /tmp/thermalwatch.tar.gz \
    src/ notebooks/ scripts/ requirements.txt \
    2>/dev/null || \
tar -czf /tmp/thermalwatch.tar.gz src/ notebooks/ requirements.txt
SIZE=$(du -sh /tmp/thermalwatch.tar.gz | cut -f1)
echo "  Uploading thermalwatch.tar.gz ($SIZE)..."
rclone copy /tmp/thermalwatch.tar.gz \
    "$GDRIVE/project/" --progress
rm /tmp/thermalwatch.tar.gz
echo "  ✅ Code uploaded"

# 2. Upload train/val indexes
echo ""
echo "[2/5] Uploading train/val indexes..."
rclone copy data/indexes/ \
    "$GDRIVE/data/indexes/" \
    --progress \
    --include "*.json"
echo "  ✅ Indexes uploaded"

# 3. Upload wildfire patches + per-area indexes
echo ""
echo "[3/5] Uploading wildfire patches..."
for area_year_dir in data/wildfire/patches/*/; do
    area_year=$(basename "$area_year_dir")
    count=$(ls "$area_year_dir" 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "  Skip (empty): $area_year"
        continue
    fi
    echo "  Uploading $area_year ($count files)..."
    rclone copy "$area_year_dir" \
        "$GDRIVE/data/wildfire/patches/$area_year/" \
        --progress \
        --transfers 8 \
        --retries 10 \
        --low-level-retries 10 \
        --retries-sleep 5s
    echo "  ✅ $area_year"
done

echo "  Uploading wildfire area indexes..."
rclone copy data/wildfire/patches/ \
    "$GDRIVE/data/wildfire/indexes/" \
    --include "*_index.json" \
    --progress
echo "  ✅ Wildfire area indexes"

# 4. Upload solar patches + per-area indexes
echo ""
echo "[4/5] Uploading solar patches..."
for area_year_dir in data/solar/patches/*/; do
    area_year=$(basename "$area_year_dir")
    count=$(ls "$area_year_dir" 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "  Skip (empty): $area_year"
        continue
    fi
    echo "  Uploading $area_year ($count files)..."
    rclone copy "$area_year_dir" \
        "$GDRIVE/data/solar/patches/$area_year/" \
        --progress \
        --transfers 8 \
        --retries 10 \
        --low-level-retries 10 \
        --retries-sleep 5s
    echo "  ✅ $area_year"
done

echo "  Uploading solar area indexes..."
rclone copy data/solar/patches/ \
    "$GDRIVE/data/solar/indexes/" \
    --include "*_index.json" \
    --progress
echo "  ✅ Solar area indexes"

# 5. Summary
echo ""
echo "[5/5] Verifying upload..."
rclone size "$GDRIVE"
echo ""
echo "✅ Upload complete!"
echo "GDrive: $GDRIVE"
