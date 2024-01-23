# 这段Python代码的主要功能是创建并设置一个日志记录器(logger)。主要功能如下：
# setup_logger函数接受一个模块名(name)和一个可选的日志输出文件名(output)作为输入。
# 这个输出文件将保存所有的日志信息。如果output参数为空，那么日志信息将只会被打印在控制台，并不会被保存在文件里。
# 该函数先尝试获取一个已经存在的logger实例。如果这个logger已经存在，那么直接返回这个logger。
# 如果该logger实例不存在，那么创建一个新的logger。同时，将这个logger的日志级别设置为INFO，并禁止这个logger向其父级logger传播日志消息。
# 该函数然后创建了一个日志信息格式器，并把这个格式器应用到一个StreamHandler实例，也就是控制台输出处理器。
# 这样一来，所有的日志信息都会被这个格式化器处理并输出到控制台。
# 如果输入参数output不为空，该函数会同时将日志信息保存到一个文件中。这个文件的路径可以由output参数指定。
# 最后，这个函数将新创建的logger实例的名字添加到全局变量logger_initialized中，然后返回这个logger实例。
# 此外，ColoredFormatter是一个自定义的日志信息格式器类，它会根据日志信息的级别改变信息的颜色。
# 总的来说，这段代码的目的就是创建一个日志记录器(logger)，这个记录器可以将日志信息以友好且易于阅读的颜色和格式打印到控制台，并可选地保存到文件中。
# import datetime
import logging
import os
import sys
import termcolor

# __all__ = ['setup_logger']意味着
# 如果其他脚本使用from your_module import *时，只有setup_logger这一个名称会被导入。
__all__ = ['setup_logger']   

logger_initialized = []   # 初始化一个空列表


def setup_logger(name, output=None):
    """
    初始化日志记录器并将其冗长程度设置为 INFO
    Initialize logger and set its verbosity level to INFO.
    Args:参数：
        output (str)：保存日志的文件名或目录。如果为空，则不保存日志文件。
        output (str): a file name or a directory to save log. If None, will not save log file.
            如果以".txt "或".log "结尾，则假定为文件名。
            If ends with ".txt" or ".log", assumed to be a file name.
            否则，日志将保存到 `output/log.txt`。
            Otherwise, logs will be saved to `output/log.txt`.
        name（str）：日志记录器的根模块名称
        name (str): the root module name of this logger

    Returns:
        logging.Logger：日志记录器
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)  # 用来获取一个logger实例的，你可以通过这个logger实例来记录你的程序运行过程中的信息。
    # 参数name通常是你定义的logger的名字。
    
    if name in logger_initialized: 
        return logger
    #  设置了日志记录的层级为INFO。日志级别指定了被追踪的信息类型和严重程度。具体来说，日志级别（从最高到最低）
    # 如下： CRITICAL> ERROR > WARNING > INFO > DEBUG > NOTSET。
    # 这行代码设置了日志级别为INFO，因此只有INFO级别及以上的日志才会被记录。低于INFO级别的 DEBUG 日志会被忽略。
    logger.setLevel(logging.INFO)  
    
    logger.propagate = False

    formatter = ("[%(asctime2)s %(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s")
    color_formatter = ColoredFormatter(formatter, datefmt="%m/%d %H:%M:%S")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(color_formatter)
    logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        os.makedirs(os.path.dirname(filename))
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter())
        logger.addHandler(fh)
    logger_initialized.append(name)
    return logger


COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt, use_color=True):
        logging.Formatter.__init__(self, fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:

            def colored(text):
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs={"bold": True},
                )

            record.levelname2 = colored("{:<7}".format(record.levelname))
            record.message2 = colored(record.msg)

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2 = termcolor.colored(asctime2, color="green")

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")
        return logging.Formatter.format(self, record)

