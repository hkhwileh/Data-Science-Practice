import  math

def isPrime_v3(n):

    if n ==1:
        return False

    if n ==2:
        return True
    if n >2 and n%2 ==0:
        return False
    max_dev = math.floor(math.sqrt(n))

    for i in range(2,1+max_dev,2):
        if n%i==0:
            return False

    return True


def isPrime_v2(n):
    """ return False is the number is not prime , return True if the number is prime"""

    if n ==1:
        return False
    max_devider = math.floor( math.sqrt(n))

    for i in range(2,1+max_devider):
        if n%i==0:
            return False
    return True

def isPrime_v1(n):

    if n ==1:
        return False

    for i in range(2,n):
        if n%i==0:
            return False
    return True

 for i in range(1,21):
    print(i,":",isPrime_v3(i))



