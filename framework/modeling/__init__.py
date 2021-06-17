from .models.tcn_baseline import TCN_BASE
from .models.tcn_blind import TCN_BLIND
from .models.tcn_DCM import TCN_DCM

ARCHITECTURES = {"TCN_BASE": TCN_BASE, 
                 "TCN_DCM": TCN_DCM,
                 "TCN_BLIND": TCN_BLIND}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
