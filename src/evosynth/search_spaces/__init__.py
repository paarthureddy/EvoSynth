"""
Search-space definitions (ConfigSpace), built from the [search_space.*] sections of
settings/search_spaces.toml. Every space is searched in the universal latent cube [0, 1]^d.
"""
from evosynth.search_spaces.tabular import get_tabular_search_space, get_xgboost_search_space
from evosynth.search_spaces.nas import get_image_nas_search_space
from evosynth.search_spaces.toy import get_dummy_search_space

__all__ = ["get_tabular_search_space", "get_xgboost_search_space",
           "get_image_nas_search_space", "get_dummy_search_space"]
