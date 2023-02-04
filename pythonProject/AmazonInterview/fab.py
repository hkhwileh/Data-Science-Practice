
def fab(n):
    if n == 1 :
        return 1
    if n == 2:
        return 1
    if n > 2:
        return fab(n-1) + fab(n -2)

for i in range(1,10):
    print(fab(i))