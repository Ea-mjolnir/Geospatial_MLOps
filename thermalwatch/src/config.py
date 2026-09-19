"""
ThermalWatch — Config Loader
==============================
Loads API keys from .env file.
Never hardcode keys in pipeline scripts.

Usage:
  from src.config import get_config
  cfg = get_config()
  firms_key = cfg['FIRMS_API_KEY']
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def get_config() -> dict:
    """
    Load API keys from .env file.
    Raises if .env not found or required keys missing.
    """
    # Find .env — look in project root
    env_path = Path(__file__).parent.parent / '.env'

    if not env_path.exists():
        raise FileNotFoundError(
            f'.env file not found: {env_path}\n'
            f'Create it with your API keys:\n'
            f'  FIRMS_API_KEY=your_key\n'
            f'  NREL_API_KEY=your_key\n'
            f'  EIA_API_KEY=your_key\n'
            f'  ERA5_KEY=your_key\n'
        )

    config = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()

    # Validate required keys
    required = [
        'FIRMS_API_KEY',
        'NREL_API_KEY',
        'EIA_API_KEY',
        'ERA5_KEY',
        'THERMALWATCH_EMAIL',
    ]
    missing = [k for k in required if not config.get(k)]

    if missing:
        raise ValueError(
            f'Missing API keys in .env: {missing}\n'
            f'Add them to {env_path}'
        )

    log.info('✅ Config loaded successfully')
    return config


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    cfg = get_config()
    for k, v in cfg.items():
        print(f'{k}: {v[:8]}...')
