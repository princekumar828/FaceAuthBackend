import multiprocessing

bind = "0.0.0.0:10000"
workers = 1  # Free tier has limited resources
threads = 2
timeout = 120  # Reduced timeout for free tier
worker_class = 'gthread'
max_requests = 500
max_requests_jitter = 100
