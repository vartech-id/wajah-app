from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PresetConfig:
    target_L: float
    max_delta_L: float
    smooth_strength: float
    eye_smooth_strength: float
    glow_strength: float
    saturation_boost: float
    hydration_highlight: float
    wrinkle_soften: float
    detail_mix: float
    unsharp_amount: float
    unsharp_radius: float
    edge_enhance_mix: float


PRESET_CONFIGS: Dict[str, PresetConfig] = {
    "cerah": PresetConfig(
        target_L=165.0,
        max_delta_L=16.0,
        smooth_strength=0.24,
        eye_smooth_strength=0.18,
        glow_strength=0.10,
        saturation_boost=1.06,
        hydration_highlight=0.0,
        wrinkle_soften=0.0,
        detail_mix=0.14,
        unsharp_amount=0.06,
        unsharp_radius=1.2,
        edge_enhance_mix=0.65,
    ),
    "lembab": PresetConfig(
        target_L=158.0,
        max_delta_L=18.0,
        smooth_strength=0.47,
        eye_smooth_strength=0.40,
        glow_strength=0.18,
        saturation_boost=1.10,
        hydration_highlight=0.25,
        wrinkle_soften=0.0,
        detail_mix=0.10,
        unsharp_amount=0.05,
        unsharp_radius=1.0,
        edge_enhance_mix=0.55,
    ),
    "kerutan": PresetConfig(
        target_L=153.0,
        max_delta_L=12.0,
        smooth_strength=0.38,
        eye_smooth_strength=0.72,
        glow_strength=0.05,
        saturation_boost=1.02,
        hydration_highlight=0.07,
        wrinkle_soften=2.00,
        detail_mix=0.08,
        unsharp_amount=0.04,
        unsharp_radius=1.0,
        edge_enhance_mix=0.45,
    ),
}

VALID_PRESETS = set(PRESET_CONFIGS.keys())

__all__ = ["PresetConfig", "PRESET_CONFIGS", "VALID_PRESETS"]
