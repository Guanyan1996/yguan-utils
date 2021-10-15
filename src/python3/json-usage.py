import gzip
import json
from pathlib import Path
from typing import List, Dict, Tuple, Hashable, Any

import brotli


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


# brotli的压缩速度很慢，但是解压速度比gzip快很多，动态文件压缩的话，用gzip，静态文件用brotli
def compress_json_gzip(jsonfilename):
    data = {}
    with gzip.open(jsonfilename, 'wt', encoding='UTF-8') as zipfile:
        json.dump(data, zipfile)

    with gzip.open(jsonfilename, 'rt', encoding='UTF-8') as zipfile:
        my_object = json.load(zipfile)


def compress_json_brotli(jsonfilename):
    data = json.dumps(json.load(open(jsonfilename, 'r')), separators=(',', ':')).encode('utf-8')
    with open("test.BR", 'wb') as f:
        content = brotli.compress(data, quality=11)
        f.write(content)
