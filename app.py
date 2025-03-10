import sys
import os
from alpha import UnifiedApp
from config import cfg
from logger import setup_logging, log_error

def main():
    setup_logging()
    log_error("Starting application from app.py")
    
    # Set profile_duration to None for normal operation, or a number (seconds) for profiling
    app = UnifiedApp(profile_duration=None)  # Change value for profiling
    cfg.set_restart_callback(app.restart)
    
    try:
        app.run()
    except KeyboardInterrupt:
        app.cleanup()
    except Exception as e:
        log_error("Unexpected error in app execution", e)
        app.cleanup()
    finally:
        os._exit(0)

if __name__ == "__main__":
    main()