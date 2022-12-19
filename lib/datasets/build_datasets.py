import sys

def build_train(cfg, mode='train'):
    if cfg.type == 'scarf':
        from .scarf_data import NerfDataset
    return NerfDataset(cfg, mode=mode)
