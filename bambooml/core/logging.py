import logging
import sys

def get_logger(name: str, stdout: bool = True, filename: str | None = None) -> logging.Logger:
    """获取日志记录器

    Args:
        name (str): 日志记录器名称
        stdout (bool, optional): 是否将日志输出到标准输出. Defaults to True.
        filename (str | None, optional): 日志文件名. Defaults to None.

    Returns:
        logging.Logger: 日志记录器
    """

    logger = logging.getLogger(name)  # 获取日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
    logger.handlers = []  # 清空已有的处理器
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')  # 日志格式，包含时间、日志级别和消息
    if stdout:
        sh = logging.StreamHandler(sys.stdout)  # 创建标准输出处理器
        sh.setFormatter(fmt)  # 设置标准输出处理器的格式
        logger.addHandler(sh)  # 添加标准输出处理器
    if filename:
        fh = logging.FileHandler(filename)  # 创建文件处理器
        fh.setFormatter(fmt)  # 设置文件处理器的格式
        logger.addHandler(fh)  # 添加文件处理器
    return logger
