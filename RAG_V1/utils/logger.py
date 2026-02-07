import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str):
    """
    配置全局日志句柄
    """
    # 1. 创建日志目录
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "rag_system.log"

    # 2. 创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 如果已经有 handler 则不再添加（防止重复打印）
    if not logger.handlers:
        # 3. 定义格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )

        # 4. 文件输出 - 轮转（每个文件10MB，最多保留5个）
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # 5. 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger