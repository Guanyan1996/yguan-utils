import subprocess

from loguru import logger


def shell_run(command):
    # universal_newlines会把b'转成'utf-8'-str
    with subprocess.Popen(command, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, encoding='utf-8') as p:
        for line in p.stdout:
            logger.info(line.rstrip())
        # 直接返回returnCode这块会返回None,需要wait等待拿到返回值
        ret_code = p.wait()
    if ret_code:
        logger.error(f"exit {ret_code}")
        raise subprocess.CalledProcessError(ret_code, command)
    else:
        logger.info(f"exit {ret_code}")
    return ret_code
