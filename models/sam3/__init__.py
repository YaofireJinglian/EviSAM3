# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

try:
    from models.model_builder import build_sam3_image_model
except (ImportError, ModuleNotFoundError):
    build_sam3_image_model = None  # Will be imported when needed

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model"]
