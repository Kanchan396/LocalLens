import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from indexer import build_or_update_index, DOCS_DIR

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            build_or_update_index()

    def on_modified(self, event):
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
            build_or_update_index()

    def on_deleted(self, event):
        if not event.is_directory:
            print(f"File deleted: {event.src_path}")
            build_or_update_index()

def start_watcher():
    DOCS_DIR.mkdir(exist_ok=True)
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, str(DOCS_DIR), recursive=True)
    observer.start()
    print(f"Watching folder: {DOCS_DIR.resolve()}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    build_or_update_index()
    start_watcher()