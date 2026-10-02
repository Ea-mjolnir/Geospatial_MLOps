"""
GeoFoundationHub — Model Router
=================================
Implements advanced serving patterns:
  1. A/B Routing    — split traffic between models
  2. Shadow Deploy  — run challenger silently
  3. Canary Rollout — gradual traffic shift
  4. Multi-tenant   — route by API key/tenant
"""

import random
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from models.base import InferenceInput, InferenceOutput
from src.routing.intelligent_router import select_best_model

log = logging.getLogger(__name__)


class RoutingDecision:
    """Records how a request was routed."""
    def __init__(
        self,
        request_id:   str,
        pattern:      str,
        champion_id:  str,
        challenger_id: Optional[str] = None,
        tenant_id:    Optional[str] = None,
    ):
        self.request_id    = request_id
        self.pattern       = pattern
        self.champion_id   = champion_id
        self.challenger_id = challenger_id
        self.tenant_id     = tenant_id
        self.timestamp     = datetime.utcnow()


class RoutingConfig:
    """Configuration for routing patterns."""
    def __init__(self):
        # A/B testing
        self.ab_enabled     = False
        self.ab_split       = 0.5        # 50% to challenger
        self.ab_champion_id = None
        self.ab_challenger_id = None

        # Shadow deployment
        self.shadow_enabled    = False
        self.shadow_champion_id   = None
        self.shadow_challenger_id = None

        # Canary rollout
        self.canary_enabled    = False
        self.canary_weight     = 0.05    # 5% to canary
        self.canary_model_id   = None
        self.baseline_model_id = None

        # Multi-tenant routing
        self.tenant_routes: Dict[str, str] = {}

        # Default model
        self.default_model_id = None


class ModelRouter:
    """
    Routes inference requests to appropriate models
    based on configured serving patterns.

    Patterns (in priority order):
    1. Tenant routing  (if tenant_id matches)
    2. Canary rollout  (if enabled)
    3. A/B testing     (if enabled)
    4. Shadow deploy   (always uses champion)
    5. Default model   (fallback)
    """

    def __init__(self):
        self.config   = RoutingConfig()
        self._adapters: Dict[str, object] = {}
        self._lock    = threading.Lock()
        self._request_count = 0
        self._routing_log: List[RoutingDecision] = []

    def register_adapter(self, model_id: str, adapter) -> None:
        """Register a loaded model adapter."""
        with self._lock:
            self._adapters[model_id] = adapter
            if self.config.default_model_id is None:
                self.config.default_model_id = model_id
            log.info(f"Router: registered adapter {model_id}")

    def configure_ab(
        self,
        champion_id:   str,
        challenger_id: str,
        split:         float = 0.5,
    ) -> None:
        """
        Configure A/B testing.
        split: fraction of traffic to challenger [0-1]
        """
        self.config.ab_enabled      = True
        self.config.ab_champion_id  = champion_id
        self.config.ab_challenger_id = challenger_id
        self.config.ab_split        = split
        log.info(
            f"A/B configured: {champion_id} vs {challenger_id} "
            f"({int(split*100)}% challenger)"
        )

    def configure_shadow(
        self,
        champion_id:   str,
        challenger_id: str,
    ) -> None:
        """
        Configure shadow deployment.
        Challenger runs silently — results logged only.
        """
        self.config.shadow_enabled      = True
        self.config.shadow_champion_id  = champion_id
        self.config.shadow_challenger_id = challenger_id
        log.info(
            f"Shadow configured: {champion_id} (champion) "
            f"+ {challenger_id} (shadow)"
        )

    def configure_canary(
        self,
        baseline_id: str,
        canary_id:   str,
        weight:      float = 0.05,
    ) -> None:
        """
        Configure canary rollout.
        weight: fraction of traffic to canary [0-1]
        """
        self.config.canary_enabled    = True
        self.config.baseline_model_id = baseline_id
        self.config.canary_model_id   = canary_id
        self.config.canary_weight     = weight
        log.info(
            f"Canary configured: {canary_id} at "
            f"{int(weight*100)}% traffic"
        )

    def add_tenant_route(self, tenant_id: str, model_id: str) -> None:
        """Route specific tenant to specific model."""
        self.config.tenant_routes[tenant_id] = model_id
        log.info(f"Tenant route: {tenant_id} → {model_id}")

    def route(
        self,
        inputs:     InferenceInput,
        tenant_id:  Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[InferenceOutput, RoutingDecision]:
        """
        Route request to appropriate model.
        Returns (output, routing_decision).
        """
        import uuid
        request_id = request_id or str(uuid.uuid4())[:8]
        self._request_count += 1

        # 1. Tenant routing — use intelligent routing
        # Tenant preferred model is a hint but intelligent
        # routing picks best model for actual input data
        if tenant_id and tenant_id in self.config.tenant_routes:
            from src.registry.registry import get_registry
            registry_models = get_registry().get_active_models()
            model_id = select_best_model(
                inputs=inputs,
                adapters=self._adapters,
                registry_models=registry_models,
            )
            if not model_id:
                model_id = self.config.tenant_routes[tenant_id]
            decision = RoutingDecision(
                request_id=request_id,
                pattern="intelligent_tenant",
                champion_id=model_id,
                tenant_id=tenant_id,
            )
            output = self._run_inference(model_id, inputs)
            self._log_decision(decision)
            return output, decision

        # 2. Canary rollout
        if self.config.canary_enabled:
            if random.random() < self.config.canary_weight:
                model_id = self.config.canary_model_id
                decision = RoutingDecision(
                    request_id=request_id,
                    pattern="canary",
                    champion_id=model_id,
                    tenant_id=tenant_id,
                )
                output = self._run_inference(model_id, inputs)
                self._log_decision(decision)
                return output, decision
            else:
                model_id = self.config.baseline_model_id
                decision = RoutingDecision(
                    request_id=request_id,
                    pattern="canary_baseline",
                    champion_id=model_id,
                    tenant_id=tenant_id,
                )
                output = self._run_inference(model_id, inputs)
                self._log_decision(decision)
                return output, decision

        # 3. A/B testing
        if self.config.ab_enabled:
            if random.random() < self.config.ab_split:
                model_id = self.config.ab_challenger_id
                pattern  = "ab_challenger"
            else:
                model_id = self.config.ab_champion_id
                pattern  = "ab_champion"
            decision = RoutingDecision(
                request_id=request_id,
                pattern=pattern,
                champion_id=model_id,
                tenant_id=tenant_id,
            )
            output = self._run_inference(model_id, inputs)
            self._log_decision(decision)
            return output, decision

        # 4. Shadow deployment
        if self.config.shadow_enabled:
            champion_id   = self.config.shadow_champion_id
            challenger_id = self.config.shadow_challenger_id
            decision = RoutingDecision(
                request_id=request_id,
                pattern="shadow",
                champion_id=champion_id,
                challenger_id=challenger_id,
                tenant_id=tenant_id,
            )
            # Run champion → return to user
            output = self._run_inference(champion_id, inputs)
            # Run challenger silently in background
            threading.Thread(
                target=self._shadow_inference,
                args=(challenger_id, inputs, request_id),
                daemon=True,
            ).start()
            self._log_decision(decision)
            return output, decision

        # 5. Intelligent routing — pick best model for input
        from src.registry.registry import get_registry
        registry_models = get_registry().get_active_models()
        model_id = select_best_model(
            inputs=inputs,
            adapters=self._adapters,
            registry_models=registry_models,
        )
        if not model_id:
            model_id = self.config.default_model_id

        decision = RoutingDecision(
            request_id=request_id,
            pattern="intelligent",
            champion_id=model_id,
            tenant_id=tenant_id,
        )
        output = self._run_inference(model_id, inputs)
        self._log_decision(decision)
        return output, decision

    def _run_inference(
        self, model_id: str, inputs: InferenceInput
    ) -> InferenceOutput:
        """Run inference on a specific model."""
        adapter = self._adapters.get(model_id)
        if not adapter:
            raise ValueError(f"No adapter registered for model_id: {model_id}")
        return adapter.timed_predict(inputs)

    def _shadow_inference(
        self, model_id: str, inputs: InferenceInput, request_id: str
    ) -> None:
        """Run shadow inference silently — log result only."""
        try:
            output = self._run_inference(model_id, inputs)
            log.info(
                f"SHADOW [{request_id}] "
                f"model={model_id} "
                f"latency={output.latency_ms}ms "
                f"emb_dim={len(output.embedding or [])}"
            )
        except Exception as e:
            log.error(f"SHADOW [{request_id}] failed: {e}")

    def _log_decision(self, decision: RoutingDecision) -> None:
        """Log routing decision."""
        self._routing_log.append(decision)
        log.info(
            f"ROUTE [{decision.request_id}] "
            f"pattern={decision.pattern} "
            f"model={decision.champion_id} "
            f"tenant={decision.tenant_id}"
        )

    def stats(self) -> Dict:
        """Routing statistics."""
        patterns = {}
        for d in self._routing_log:
            patterns[d.pattern] = patterns.get(d.pattern, 0) + 1
        return {
            "total_requests": self._request_count,
            "registered_models": list(self._adapters.keys()),
            "routing_patterns": patterns,
            "ab_enabled":     self.config.ab_enabled,
            "shadow_enabled": self.config.shadow_enabled,
            "canary_enabled": self.config.canary_enabled,
            "canary_weight":  self.config.canary_weight,
            "tenant_routes":  self.config.tenant_routes,
        }


# Global singleton
_router = None

def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
