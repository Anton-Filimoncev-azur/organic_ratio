from typing import Callable, Dict

from organic_ratio.core.preprocessing.installs import build_installs_features
from organic_ratio.core.preprocessing.iap import build_iap_features
from organic_ratio.core.preprocessing.ads import build_ads_features
from organic_ratio.core.preprocessing.costs import build_costs_features
from organic_ratio.core.preprocessing.sessions import build_sessions_features
from organic_ratio.core.preprocessing.personal import build_personal_features
from organic_ratio.core.preprocessing.devices import build_devices_features


PREPROCESSORS: Dict[str, Callable] = {
    "installs": build_installs_features,
    "ads": build_ads_features,
    "iap": build_iap_features,
    "sessions": build_sessions_features,
    "personal": build_personal_features,
    "costs": build_costs_features,
    "devices": build_devices_features,
}
