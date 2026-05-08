from omegaconf import OmegaConf
from pathlib import Path
import os


CONFIG_OVERRIDE_ENV = "CONFIG_OVERRIDE_PATH"


def load_config(env: str = "base"):
    root = Path(__file__).resolve().parents[3]

    globals_cfg = OmegaConf.load(root / "conf/base/globals.yml")
    data_cfg = OmegaConf.load(root / "conf/base/data.yml")
    params_cfg = OmegaConf.load(root / "conf/base/parameters.yml")
    base_cfg = OmegaConf.merge(globals_cfg, data_cfg, params_cfg)

    override_path = os.getenv(CONFIG_OVERRIDE_ENV)
    if override_path:
        override_cfg = OmegaConf.load(override_path)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg

    OmegaConf.resolve(cfg)
    return cfg
