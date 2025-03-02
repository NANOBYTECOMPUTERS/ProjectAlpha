import logging
import logging.handlers
import queue

class CudaLogFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ("dealloc: cuMemFree_v2" in msg or 
                   "add pending dealloc: cuMemFree_v2" in msg or
                   "mouse_event" in msg.lower())

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.handlers.QueueHandler(queue.Queue(-1))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.addFilter(CudaLogFilter())
    logger.addHandler(handler)
    listener = logging.handlers.QueueListener(handler.queue, logging.FileHandler('app.log'))
    listener.start()

def log_error(message, exception=None):
    logging.error(f"{message}: {str(exception)}" if exception else message)