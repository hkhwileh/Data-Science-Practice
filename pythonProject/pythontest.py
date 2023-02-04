import time
import math
''' prime number can be devided by itself and by 1 only'''
def is_prime_number_v1(n):
    if n ==1:
        return False
    for x in range(2,n):
        if n % x == 0:
            return False

        return True
def is_prime_number_v2(n):
    if n ==1:
        return False
    max_divisor = math.floor(math.sqrt(n))
    for x in range(2,max_divisor+1):
        if n % x == 0:
            return False
        return True
def is_prime_number_v3(n):
    if n ==1:
        return False
    if n>2 and n%2==0:
        return False
    max_divisor = math.floor(math.sqrt(n))
    for x in range(2,max_divisor+1):
        if n % x == 0:
            return False
        return True


t0 = time.time()
for n in range(1,20):
    print(n,is_prime_number_v3(n))

t1 = time.time()
print("The time required: ", t1-t0)