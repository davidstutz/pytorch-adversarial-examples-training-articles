from common.log import log, Timer


class ProgressBar:
    """
    Simple wrapper for terminal progress.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.timer = Timer()
        """ (Timer) Timer. """

    def __call__(self, name, batch, batches, info='', width=20):
        """
        Report progress.

        :param name: epoch
        :type name: int
        :param batch: batch
        :type batch: int
        :param batches: batches
        :type batches: int
        """

        elapsed = self.timer.elapsed()

        percent = round((batch/batches)*100)
        completed = round(percent / (100/width))
        incompleted = width - round(percent / (100/width))

        if batch == batches - 1:
            log('%s: <%s> %d/%d (%dms) %s' % (
                str(name),
                ('=' * width),
                batch + 1,
                batches,
                round(elapsed * 1000),
                info,
            ), end="\n", erase=True)
        else:
            log('%s: <%s> %d/%d (%dms) %s' % (
                str(name),
                ('=' * completed + ' ' * incompleted),
                batch + 1,
                batches,
                round(elapsed * 1000),
                info,
            ), end="\n", erase=True)

        self.timer.reset()