# https://github.com/Delgan/loguru
from loguru import logger


@logger.catch
def my_function(x, y, z):
    # An error? It's caught anyway!
    return 1 / (x + y + z)


logger.info("hello,wolrd")
my_function(0, 0, 0)
