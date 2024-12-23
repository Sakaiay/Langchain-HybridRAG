import os
import logging
import datetime

def setup_logger(log_dir="logs", log_level=logging.DEBUG):
    """
    设置日志记录器
    log_dir: 日志文件夹路径
    log_level: 日志记录等级
    """
    datenow = datetime.datetime.now()
    log_name = str(datenow.year) + "-" + "{:02d}".format(datenow.month) \
+ "{:02d}".format(datenow.day) + "-" + "{:02d}".format(datenow.hour) + "-" + "{:02d}".format(datenow.minute) + ".log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_name)
    
   
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
    )
    # 创建日志记录器
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(log_level)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(log_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("测试")
    logger.debug("debug")
