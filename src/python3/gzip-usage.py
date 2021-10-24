# brotli的压缩速度很慢，但是解压速度比gzip快很多，动态文件压缩的话，用gzip，静态文件用brotli
import gzip
import json


def compress_json_gzip(jsonfilename):
    data = {}
    with gzip.open(jsonfilename, 'wt', encoding='UTF-8') as zipfile:
        json.dump(data, zipfile)

    with gzip.open(jsonfilename, 'rt', encoding='UTF-8') as zipfile:
        my_object = json.load(zipfile)
