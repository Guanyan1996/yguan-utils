import tarfile
from loguru import logger


def tar_gz(tar_gz_file_name, tar_path, arcname):
    """
    Args:
        tar_gz_file_name: 压缩成的名字和路径
        tar_path: 准备压缩的文件目录
        arcname: 压缩文件目录压缩后的目录格式
    Returns: str: tar_gz_file_name
    """
    with tarfile.open(tar_gz_file_name, "w:gz") as t:
        t.add(tar_path, arcname=arcname)
        logger.info(f"{tar_path} 压缩成功为：{tar_gz_file_name}")
    return tar_gz_file_name