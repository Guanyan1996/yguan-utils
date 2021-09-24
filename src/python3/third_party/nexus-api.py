import requests
from loguru import logger


class NexusApi(object):
    def __init__(self, user: str, passwd: str):
        self._auth = (user, passwd)

    def upload_file(self, url: str, file: str):
        """
        Args:
            url: upload的web端url全路径包含file名
            file: 本地要上传的文件路径
        Returns:
        """
        logger.info(f"ready upload to {url}")
        response = requests.put(url=url, data=open(file, "rb").read(), auth=self._auth)
        if response.status_code == 201:
            logger.info(f"upload {url} successfully")
        else:
            logger.info(response.raise_for_status())
