from collections import Counter


def numberToWord( number):
    X = ["", " One ", " Two ", " Three ", " Four ", " Five ", " Six ",
            " Seven ", " Eight ", " Nine ", " Ten ", " Eleven ", " Twelve ",
            " Thirteen ", " Fourteen ", " Fifteen ", " Sixteen ",
            " Seventeen ", " Eighteen ", " Nineteen "]
    Y = ["", "", " Twenty ", " Thirty ", " Forty ", " Fifty ",
            " Sixty ", " Seventy ", " Eighty ", " Ninety "]

    z = {1000:" Thousand", 100:" Hundred"}

    x = number
    wordNumber = ""
    while x > 0:

        if x >= 1000:
            wordNumber = wordNumber +X[x//1000] +z[1000]
            x =x%1000
            continue
        elif x >= 100:
            wordNumber = wordNumber + X[x // 100] + z[100]
            x = x % 100
            continue
        elif 100 > x > 20:
            wordNumber = wordNumber + Y[x // 10]
            x = x % 10
            continue
        elif 20 > x:
            wordNumber = wordNumber + X[x]
            x = -1
            continue

    return (wordNumber)
print(numberToWord(22903))