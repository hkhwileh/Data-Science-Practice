from functools import lru_cache

@lru_cache(maxsize=1000)
def fibonacci(n):
    if n <1:
        return 0
    if type(n) != int:
        raise
    if n ==1:
        return 1
    if n ==2:
        return 1
    if n>2:
        return fibonacci(n-1)+fibonacci(n-2)


print(0,":",fibonacci("rewyttet"))