import sys
from contextlib import contextmanager
from uuid import uuid4 as uuid

import docker


class GDocker:
    def __init__(self):
        self.client = docker.from_env()

    def login(self, username, password, registry):
        try:
            self.client.login(username=username, password=password, registry=registry)
        except Exception:
            print("docker login 失败,程序退出")
            sys.exit()

    @contextmanager
    def contains_run(self, **kwargs):
        print(kwargs)
        container = self.client.containers.run(**kwargs)
        try:
            yield container
        finally:
            container.stop()
            container.remove(force=True)


if __name__ == '__main__':
    gdocker = GDocker()
    with gdocker.contains_run(image="",
                              detach=True,
                              name=f"Gdocker_{uuid()}",
                              auto_remove=False,
                              volumes={"/xxx/xxx": {"bind": "/ssd", "mode": "rw"}},
                              tty=True,
                              runtime="nvidia",
                              working_dir="",
                              command="bash",
                              ) as c:
        # for log in c.logs(timestamps=True, stream=True):
        #     print(log)
        #  (ExecResult): A tuple of (exit_code, output)
        # if demux = True ,then print (stdout ,stderr)

        resp = c.exec_run(cmd="ls",
                          stream=True,
                          # demux=True,
                          stderr=True,
                          stdout=True,
                          workdir="")
        for log in resp[1]:
            print(log)
        # print(f"exit_code={resp[0]}")
