import json
from pathlib import Path
from typing import List, Dict, Tuple, Hashable, Any


def raise_on_duplicate_keys(ordered_pairs: List[Tuple[Hashable, Any]]) -> Dict:
    """Raise ValueError if a duplicate key exists in provided ordered list of pairs, otherwise return a dict."""
    dict_out = {}
    for key, val in ordered_pairs:
        if key in dict_out:
            raise ValueError(f'Duplicate key: {key}')
        else:
            dict_out[key] = val
    return dict_out


def check_duplicate_keys(file):
    """检查json文件中是否存在重复的key，原hook默认为update"""
    return json.load(Path(file).open('r'), object_pairs_hook=raise_on_duplicate_keys)
