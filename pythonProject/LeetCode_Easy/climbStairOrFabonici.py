n =10

one= 0
two = 1

for i in range(n):
    print(one)
    temp = one
    one = one + two
    two = temp
    print("i", i)
