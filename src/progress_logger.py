from tqdm import tqdm
import logging
import threading
from queue import Queue


class ProgressLogger:
    def __init__(self):
        self.progress_bars = {}
        self.message_queue = Queue()
        self._stop_event = threading.Event()
        self._logger_thread = threading.Thread(target=self._process_messages)
        self._logger_thread.start()

    def create_progress_bar(self, task_id, total, desc):
        self.progress_bars[task_id] = tqdm(total=total, desc=desc, leave=True)
        return task_id

    def update_progress(self, task_id, n=1):
        self.message_queue.put(("update", task_id, n))

    def close_progress_bar(self, task_id):
        self.message_queue.put(("close", task_id, None))

    def _process_messages(self):
        while not self._stop_event.is_set() or not self.message_queue.empty():
            try:
                action, task_id, value = self.message_queue.get(timeout=0.1)
                if action == "update" and task_id in self.progress_bars:
                    self.progress_bars[task_id].update(value)
                elif action == "close" and task_id in self.progress_bars:
                    self.progress_bars[task_id].close()
                    del self.progress_bars[task_id]
                self.message_queue.task_done()
            except:
                continue

    def shutdown(self):
        self._stop_event.set()
        self._logger_thread.join()
        for bar in self.progress_bars.values():
            bar.close()
        self.progress_bars.clear()


class ProgressHandler(logging.Handler):
    def __init__(self, progress_logger):
        super().__init__()
        self.progress_logger = progress_logger

    def emit(self, record):
        try:
            msg = self.format(record)
            pass
        except Exception:
            self.handleError(record)


def setup_progress_logging():
    progress_logger = ProgressLogger()
    handler = ProgressHandler(progress_logger)
    logging.getLogger().addHandler(handler)
    return progress_logger
