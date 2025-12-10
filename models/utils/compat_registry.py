"""Compatibility shim for MMCV 1.x -> 2.x registry migration.

In MMCV 2.x / MMEngine, the old registries like ATTENTION, TRANSFORMER_LAYER,
TRANSFORMER_LAYER_SEQUENCE, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING have been
removed. This module provides drop-in replacements using mmengine.registry.Registry.

Usage:
    from projects.mmdet3d_plugin.models.utils.compat_registry import (
        ATTENTION,
        TRANSFORMER_LAYER,
        TRANSFORMER_LAYER_SEQUENCE,
        FEEDFORWARD_NETWORK,
        POSITIONAL_ENCODING,
    )
"""

from mmengine.registry import Registry

# Create standalone registries that mirror the old MMCV 1.x behavior
ATTENTION = Registry('attention', locations=['projects.mmdet3d_plugin.bevformer.modules'])
TRANSFORMER_LAYER = Registry('transformer_layer', locations=['projects.mmdet3d_plugin.bevformer.modules'])
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer_layer_sequence', locations=['projects.mmdet3d_plugin.bevformer.modules'])
FEEDFORWARD_NETWORK = Registry('feedforward_network')
POSITIONAL_ENCODING = Registry('positional_encoding')
