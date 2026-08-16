from concurrent.futures import ThreadPoolExecutor

MAX_THREADS = 5

def run_in_threads(func, iterable, max_workers=MAX_THREADS):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in iterable]
        return [f.result() for f in futures]
