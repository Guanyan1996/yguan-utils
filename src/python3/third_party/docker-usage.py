"""
!/usr/bin/python3.7
-*- coding: UTF-8 -*-
Author:https://github.com/Guanyan1996
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
import docker
import docker.api.build
from loguru import logger


# pip3 install docker-py

class Dock(object):

    def __init__(self):
        self._cli = docker.from_env()

    def login(self, username: str, password: str, registry: str, reauth=True) -> dict:
        resp = self._cli.login(username=username, password=password, registry=registry, reauth=reauth)
        logger.info(f"docker login: {resp}")
        return resp

    def build(self, dockerfile: str, repository: str, path: str) -> str:
        logger.info(f"ready to build {repository}")
        docker.api.build.process_dockerfile = lambda dockerfile, path: ('Dockerfile', dockerfile)
        _, logs = self._cli.images.build(path=path, dockerfile=dockerfile, tag=repository, quiet=True)
        for chunk in logs:
            if 'stream' in chunk and chunk['stream'] != '\n':
                for line in chunk['stream'].splitlines():
                    logger.info(f"docker build: {line}")
        return repository

    def push(self, repository: str) -> str:
        logger.info(f"ready to push {repository}")
        for line in self._cli.images.push(repository=repository, stream=True, decode=True):
            logger.info(f"docker push: {line}")
        return repository

    def inspect(self, repository: str):
        logger.info(f"docker inspect: {self._cli.images.get(repository)}")
