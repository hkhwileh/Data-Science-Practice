
fibonacci_cache = {}
def fib(n):

    if n in fibonacci_cache:
        return fibonacci_cache[n]
    value = 0
    if n == 1:
        value=  1
    if n == 2:
        value= 1
    if n > 2:
        value= fib(n-1) + fib(n-2)
    fibonacci_cache[n]= value

    return value


for i in range(1,10):
    print(i , " : " ,fib(i))