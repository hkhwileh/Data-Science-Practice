def primes(n):
    primeslist = [2]
    for i in range(2, n):
        p = 1
        for j in primeslist:

            if (i % j) == 0:
                p = 0
                break
        if p == 1:
            primeslist.append(i)
    return primeslist


primeslist = primes(66)
print(primeslist)