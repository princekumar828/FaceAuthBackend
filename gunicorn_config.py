import os
import multiprocessing

port = os.environ.get('PORT', 10000)
bind = f"0.0.0.0:{port}"
workers = 1
threads = 2
timeout = 120
worker_class = 'gthread'
max_requests = 25  # Reduced from 50
max_requests_jitter = 5  # Reduced from 10
worker_tmp_dir = '/tmp'  # Use temporary directory
preload_app = False  # Disable preloading
max_worker_lifetime = 3600  # 1 hour max worker lifetime
