"""
GeoFoundationHub — Multi-Tenant Auth
======================================
API key based authentication.
Each tenant gets a unique API key
that maps to their model routing config.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

log = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class Tenant(BaseModel):
    """A registered tenant/organization."""
    tenant_id:   str
    name:        str
    api_key:     str
    model_id:    Optional[str] = None  # preferred model
    rate_limit:  int = 1000            # requests per day
    created_at:  datetime = datetime.utcnow()
    active:      bool = True


class TenantManager:
    """
    Manages tenants and their API keys.
    In production backed by PostgreSQL.
    """

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._key_to_tenant: Dict[str, str] = {}
        self._seed_tenants()

    def _seed_tenants(self):
        """Pre-register demo tenants."""

        self.register(Tenant(
            tenant_id="fire-agency",
            name="California Fire Department",
            api_key="fire-agency-key-2026",
            model_id=None,  # set after registry loads
            rate_limit=5000,
        ))
        self.register(Tenant(
            tenant_id="nasa-earth",
            name="NASA Earth Science Division",
            api_key="nasa-earth-key-2026",
            model_id=None,
            rate_limit=10000,
        ))
        self.register(Tenant(
            tenant_id="agri-co",
            name="AgroCrop Analytics",
            api_key="agri-co-key-2026",
            model_id=None,
            rate_limit=2000,
        ))
        self.register(Tenant(
            tenant_id="public",
            name="Public API",
            api_key="public-key-2026",
            model_id=None,
            rate_limit=100,
        ))
        log.info(f"✅ TenantManager: {len(self._tenants)} tenants registered")

    def register(self, tenant: Tenant) -> Tenant:
        """Register a new tenant."""
        self._tenants[tenant.tenant_id]     = tenant
        self._key_to_tenant[tenant.api_key] = tenant.tenant_id
        log.info(f"Tenant registered: {tenant.name} [{tenant.tenant_id}]")
        return tenant

    def get_by_key(self, api_key: str) -> Optional[Tenant]:
        """Look up tenant by API key."""
        tenant_id = self._key_to_tenant.get(api_key)
        if not tenant_id:
            return None
        return self._tenants.get(tenant_id)

    def get(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def list(self):
        """List all tenants."""
        return list(self._tenants.values())

    def create_api_key(self, tenant_id: str, name: str) -> Tenant:
        """Create a new tenant with generated API key."""
        api_key = f"{tenant_id}-{str(uuid.uuid4())[:8]}"
        tenant  = Tenant(
            tenant_id=tenant_id,
            name=name,
            api_key=api_key,
        )
        self.register(tenant)
        return tenant

    def set_model(self, tenant_id: str, model_id: str) -> bool:
        """Assign a model to a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        data = tenant.dict()
        data['model_id'] = model_id
        self._tenants[tenant_id] = Tenant(**data)
        log.info(f"Tenant {tenant_id} → model {model_id}")
        return True

    def summary(self) -> Dict:
        return {
            "total_tenants": len(self._tenants),
            "tenants": [
                {
                    "id":    t.tenant_id,
                    "name":  t.name,
                    "model": t.model_id,
                    "limit": t.rate_limit,
                }
                for t in self._tenants.values()
            ]
        }


# Global singleton
_tenant_manager = None

def get_tenant_manager() -> TenantManager:
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager


async def verify_api_key(
    api_key: str = Security(API_KEY_HEADER)
) -> Tenant:
    """
    FastAPI dependency — verifies API key
    and returns tenant. Raises 401 if invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="X-API-Key header required",
        )
    manager = get_tenant_manager()
    tenant  = manager.get_by_key(api_key)
    if not tenant:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    if not tenant.active:
        raise HTTPException(
            status_code=403,
            detail="Tenant account inactive",
        )
    return tenant
