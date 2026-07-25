"""
Feast Feature Services — GeoAI MLOps Project 1
===============================================
Feature services define WHAT the training and inference code requests.
Using named services (not raw feature views) ensures training and
inference always request exactly the same features — no drift.

Usage in training:
    store = FeatureStore(repo_path="feature_store/")
    training_df = store.get_historical_features(
        entity_df=pairs_df,
        features=model_input_service,
    ).to_df()

Usage in inference:
    store = FeatureStore(repo_path="feature_store/")
    features = store.get_online_features(
        features=model_input_service.feature_refs,
        entity_rows=[{"patch_id": "S2A_32UNE_20260101__patch_0256_0256"}],
    ).to_dict()
"""
from feast import FeatureService
from feature_views import sentinel2_patch_features, sentinel2_change_label_features

# ── Service 1: what the MODEL receives as input (34 features) ────
model_input_service = FeatureService(
    name="model_input_features",
    features=[sentinel2_patch_features],
    description=(
        "The 34 tabular features fed into the Siamese Prithvi model. "
        "Used identically at training time (get_historical_features) "
        "and inference time (get_online_features) — no train/serve skew."
    ),
)

# ── Service 2: what the LABEL COMPUTATION uses (3 excluded features)
label_computation_service = FeatureService(
    name="change_label_features",
    features=[sentinel2_change_label_features],
    description=(
        "The 3 excluded features for computing the weak supervision target: "
        "target = sqrt(d_ndvi² + d_ndwi² + d_ndbi²). "
        "Fetched at pairing time, never fed to the model as input."
    ),
)
