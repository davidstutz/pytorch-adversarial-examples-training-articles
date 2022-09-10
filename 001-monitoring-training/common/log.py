"""
Should be standalone to avoid loops in imports!
"""
import io
import sys
import time
import torch
import datetime
from enum import Enum


# to avoid import common.torch and the cycle resulting from it
def memory():
    """
    Get memory usage.

    :return: memory usage
    :rtype: str
    """

    index = torch.cuda.current_device()

    return '%d/%dMiB' % (
        (torch.cuda.memory_allocated(index) + torch.cuda.memory_reserved(index))/(1024*1024),
        (torch.cuda.max_memory_allocated(index) + torch.cuda.max_memory_reserved(index))/(1024*1024),
    )


class Timer:
    """
    Simple wrapper for time.clock().
    """

    def __init__(self, timer=time.process_time):
        """
        Initialize and start timer.
        """

        self.timer = timer
        """ (callable) Timer. """

        self.start = self.timer()
        """ (float) Seconds. """

    def reset(self):
        """
        Reset timer.
        """

        self.start = self.timer()

    def elapsed(self):
        """
        Get elapsed time in seconds.

        :return: elapsed time in seconds
        :rtype: float
        """

        return (self.timer() - self.start)


def elapsed(function):
    """
    Time a function call.

    :param function: function to call
    :type function: callable
    """

    assert callable(function)

    timer = Timer()
    results = function()
    time = timer.elapsed()

    if results is None:
        results = time
    elif isinstance(results, tuple):
        results = tuple(list(results) + [time])
    else:
        results = (results, time)

    return results


class LogLevel(Enum):
    """
    Defines log level.
    """

    INFO = 1
    WARNING = 2
    ERROR = 3


class Log:
    """
    Simple singleton log implementation with different drivers.
    """

    instance = None
    """ (Log) Log instance. """

    def __init__(self):
        """
        Constructor.
        """

        self.files = dict()
        """ ([file]) Files to write log to (default is sys.stdout). """

        self.verbose = LogLevel.INFO
        """ (LogLevel) Verbosity level. """

        self.silent = False
        """ (bool) Whether to be silent in the console. """

        self.L = 0
        """ (int) Current line. """

    class LogMessage:
        """
        Wrap a simple message.
        """

        def __init__(self, message, level=LogLevel.INFO, end="\n", erase=False):
            """
            Constructor.

            :param message: message
            :type message: str
            :param level: level
            :type level: int
            :param end: end
            :type end: str
            :param erase: whether to erase line
            :type erase: bool
            """

            self.message = message
            """ (str) Message. """

            self.level = level
            """ (LogLevel) Level. """

            self.end = end
            """ (str) End of line. """

            self.timer = Timer()
            """ (Timer) Timer. """

            self.erase = erase
            """ (bool) Erase line. """

            #              0           1 = INFO    2 = WARNING 3 = ERROR
            self.colors = ['\033[94m', '\033[94m', '\033[93m', '\033[91m\033[1m']
            """ ([str]) Level colors. """

        def timestamp(self):
            """
            Print timestamp.

            :return: date and time
            :rtype: str
            """

            dt = datetime.datetime.now()
            return '[%s|%s] ' % (dt.strftime('%d%m%y%H%M%S'), memory())

        def dispatch(self):
            """
            Simply write log message.
            """

            files = Log.get_instance()._files()
            for key in files.keys():
                files[key].write(self.timestamp())
                files[key].write(str(self.message))
                files[key].write(str(self.end))
                files[key].flush()

            if not Log.get_instance().silent:
                if self.erase:
                    # TODO
                    pass
                sys.stdout.write(self.colors[self.level.value])
                sys.stdout.write(self.timestamp())
                sys.stdout.write(str(self.message))
                sys.stdout.write('\033[0m')
                sys.stdout.write(str(self.end))
                sys.stdout.flush()

    def __del__(self):
        """
        Close files.
        """

        keys = list(self.files.keys()) # Force a list instead of an iterator!
        for key in keys:
            if isinstance(self.files[key], io.TextIOWrapper):
                self.files[key].close()
                del self.files[key]

    def attach(self, file):
        """
        Attach a file to write to.
        :param file: log file
        :type file: file
        """

        self.files[file.name] = file

    def detach(self, key):
        """
        Detach a key.

        :param key: log file name
        :type key: str
        """

        assert isinstance(key, str)
        if key in self.files.keys():
            if isinstance(self.files[key], io.TextIOWrapper):
                self.files[key].close()
                del self.files[key]

    def _files(self):
        """
        Get files.

        :return: files
        :rtype: [File]
        """

        return self.files

    @staticmethod
    def get_instance():
        """
        Get current log instance, simple singleton.
        :return: log
        :rtype: Log
        """

        if Log.instance is None:
            Log.instance = Log()

        return Log.instance

    def verbose(self, level=LogLevel.INFO):
        """
        Sets the log verbostiy.

        :param level: minimum level to report
        :return: LogLevel
        """

        self.verbose = level

    def log(self, message, level=LogLevel.INFO, end="\n", erase=False):
        """
        Log a message.

        :param message: message or variable to log
        :type message: mixed
        :param level: level, i.e. color
        :type level: LogColor
        :param end: whether to use carriage return
        :type end: str
        :param erase: whether to erase line
        :type erase: bool
        """

        if level.value >= self.verbose.value:
            Log.LogMessage(message, level, end, erase=erase).dispatch()
        if end == "\n":
            self.L += 1


def log(message, level=LogLevel.INFO, end="\n", erase=False):
    """
    Quick access to logger instance.

    :param message: message or variable to log
    :type message: mixed
    :param level: level, i.e. color
    :type level: LogColor
    :param end: whether to use carriage return
    :type end: str
    :param erase: whether to erase line
    :type erase: bool
    """

    Log.get_instance().log(message, level=level, end=end, erase=erase)
