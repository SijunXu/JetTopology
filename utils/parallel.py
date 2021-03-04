import os
import multiprocessing
from functools import partial, wraps
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def make_parallel(func, n_jobs=-1, **kwargs):
    r'''
    parallized for heavy computation for a function `func` with arguments `kwargs`. 

    example: 
        ``
        def func(x, y):
            return x ** 2 + y

        result = make_parallel(func, y=1)(lst)
        ``
    '''

    if n_jobs == -1:
        cores = multiprocessing.cpu_count()
        
    @wraps(func)
    def wrapper(lst):
        with poolcontext(processes=cores) as pool:
            result = pool.map(partial(func, **kwargs), lst)
        return result

    return wrapper
