"""
GeoFoundationHub — Model Loader
=================================
Loads all registered foundation model
adapters and wires them to the router
at application startup.

Uses lazy loading — models load on first
request to avoid slow startup times.
"""

import logging
from typing import Dict, Optional
from src.registry.registry import get_registry, ModelRegistry
from src.routing.router import get_router, ModelRouter
from src.auth.auth import get_tenant_manager

log = logging.getLogger(__name__)


def get_adapter_class(model_name: str):
    """Return adapter class for a given model name."""
    adapters = {
        "Prithvi-EO-2.0-300M": ("models.prithvi.adapter", "PrithviAdapter"),
        "SatMAE":               ("models.satmae.adapter",  "SatMAEAdapter"),
        "RemoteCLIP":           ("models.remoteclip.adapter", "RemoteCLIPAdapter"),
    }
    if model_name not in adapters:
        return None
    module_path, class_name = adapters[model_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_all_models(
    registry: Optional[ModelRegistry] = None,
    router:   Optional[ModelRouter]   = None,
) -> Dict[str, object]:
    """
    Load all active models from registry
    and register their adapters with the router.
    Returns dict of {model_id: adapter}.
    """
    registry = registry or get_registry()
    router   = router   or get_router()

    models   = registry.get_active_models()
    adapters = {}

    log.info(f"Loading {len(models)} active models...")

    for model in models:
        adapter_cls = get_adapter_class(model.name)
        if not adapter_cls:
            log.warning(f"No adapter for {model.name} — skipping")
            continue

        try:
            log.info(f"Loading {model.name} [{model.model_id}]...")
            adapter = adapter_cls(model_id=model.model_id)
            adapter.load(checkpoint_path=model.checkpoint_path)
            router.register_adapter(model.model_id, adapter)
            adapters[model.model_id] = adapter
            log.info(f"✅ {model.name} loaded and registered")
        except Exception as e:
            import traceback
            log.error(f"❌ Failed to load {model.name}: {e}")
            log.error(traceback.format_exc())

    # Set default model (first active)
    if adapters and not router.config.default_model_id:
        first_id = list(adapters.keys())[0]
        router.config.default_model_id = first_id
        log.info(f"Default model: {first_id}")

    # Wire tenant routes to model IDs
    _wire_tenant_routes(registry, router)

    log.info(f"✅ {len(adapters)} models loaded and ready")
    return adapters


def _wire_tenant_routes(registry: ModelRegistry, router: ModelRouter):
    """
    Wire default tenant→model assignments:
    fire-agency → Prithvi (thermal/optical)
    nasa-earth  → SatMAE (multispectral)
    agri-co     → RemoteCLIP (optical classification)
    """
    tm = get_tenant_manager()

    assignments = {
        "fire-agency": "Prithvi-EO-2.0-300M",
        "nasa-earth":  "SatMAE",
        "agri-co":     "RemoteCLIP",
    }

    for tenant_id, model_name in assignments.items():
        model = registry.get_by_name(model_name)
        if model and model.model_id in router._adapters:
            router.add_tenant_route(tenant_id, model.model_id)
            tm.set_model(tenant_id, model.model_id)
            log.info(f"Tenant {tenant_id} → {model_name} [{model.model_id}]")
        else:
            log.warning(f"Tenant {tenant_id}: model {model_name} not available")
