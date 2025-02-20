import os
import multiprocessing

# Always use the PORT environment variable provided by Render
port = int(os.environ.get('PORT', 8080))
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1
threads = 2
worker_class = 'gthread'
timeout = 120
keepalive = 5
worker_tmp_dir = '/tmp'
preload_app = False

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
