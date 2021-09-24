"""
!/usr/bin/python3.7
-*- coding: UTF-8 -*-
Author: https://github.com/Guanyan1996
         ┌─┐       ┌─┐
      ┌──┘ ┴───────┘ ┴──┐
      │                 │
      │       ───       │
      │  ─┬┘       └┬─  │
      │                 │
      │       ─┴─       │
      │                 │
      └───┐         ┌───┘
          │         │
          │         │
          │         │
          │         └──────────────┐
          │                        │
          │                        ├─┐
          │                        ┌─┘
          │                        │
          └─┐  ┐  ┌───────┬──┐  ┌──┘
            │ ─┤ ─┤       │ ─┤ ─┤
            └──┴──┘       └──┴──┘
                神兽保佑
                代码无BUG!

"""
import base64
import os
import urllib

import requests
from loguru import logger


def base64_2_str(base64_encode):
    """
    Args:
        base64_encode: base64编码str

    Returns: str(utf-8)

    """
    base64_decode = base64.b64decode(base64_encode).decode('utf-8')
    return base64_decode


def get_gitlab_file(gitlab_ip, project_id, git_commit, private_token,
                    filepath, output=None):
    """
    Args:
        gitlab_ip: 例如https://gitlab.xxx.cn
        project_id: 从项目的gitlab首页可以看到Project ID
        git_commit: 文件所处项目git_commit_id或者branch
        private_token: 登陆code后点击左上角头像，Perferences-Access Token - create
        filepath: 文件所处的项目下的路径
        output: 文件download 本地路径，output不存在，则不下载
    Returns: str(git file content)
    """
    quote_filepath = urllib.parse.quote(filepath, safe='', encoding=None,
                                        errors=None)
    logger.info(quote_filepath)
    interface = f"api/v4/projects/{project_id}/repository/files/{quote_filepath}?ref={git_commit}"
    gitlab_file_url = urllib.parse.urljoin(gitlab_ip, interface)
    logger.info(gitlab_file_url)
    header = {"PRIVATE-TOKEN": private_token}
    res = requests.get(headers=header, url=gitlab_file_url)
    logger.info(f"{res.status_code}, {res.json()['file_name']}")
    file_content = base64_2_str(res.json()["content"])
    logger.info(file_content)
    if output:
        with open(os.path.join(output, os.path.basename(filepath)), "w") as f:
            f.write(file_content)
    return file_content
