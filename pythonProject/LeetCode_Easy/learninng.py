from collections import Counter

number = 222
def numberToWord(self, number)
    X = ["", "One ", "Two ", "Three ", "Four ", "Five ", "Six ",
            "Seven ", "Eight ", "Nine ", "Ten ", "Eleven ", "Twelve ",
            "Thirteen ", "Fourteen ", "Fifteen ", "Sixteen ",
            "Seventeen ", "Eighteen ", "Nineteen "]
    Y = ["", "", "Twenty ", "Thirty ", "Forty ", "Fifty ",
            "Sixty ", "Seventy ", "Eighty ", "Ninety "]

    x = number
    wordNumber = ""
    while x>0:
        if x>=1000:
            thousand = x//1000
            wordNumber = X[thousand]+" thousand "
            x =x%1000
            continue
        elif x>=100:
            handred = x//100
            wordNumber = wordNumber+" "+ X[handred]+" handred "
            x =x%100
            continue
        elif 99 >= x >20:
            xx = x//10
            wordNumber =wordNumber+" "+ Y[xx]
            x = x%10
            continue
        elif 20>x:
            wordNumber = wordNumber + " " + X[x]
            x = -1

    return (wordNumber)
