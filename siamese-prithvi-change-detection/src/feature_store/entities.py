"""
Feast Entity Definitions — GeoAI MLOps Project 1
================================================
Entities are the "primary keys" that tie features together.
A Patch is the fundamental unit: one 256×256 geographic square.
"""
from feast import Entity, ValueType

# The primary entity — one unique 256×256 patch at one specific date
# patch_id format: <tile_id>__patch_<row>_<col>
# e.g. S2A_32UNE_20240920_0_L2A__patch_0256_0256
patch = Entity(
    name="patch_id",
    value_type=ValueType.STRING,
    description="Unique identifier for a 256x256 Sentinel-2 patch at a specific date",
)
