import os

# Always use the PORT environment variable provided by Render
port = int(os.environ.get('PORT', 10000))
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1
threads = 2
worker_class = 'sync'  # Change to sync for better compatibility
timeout = 120
keepalive = 5
worker_tmp_dir = '/tmp'
preload_app = False

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
