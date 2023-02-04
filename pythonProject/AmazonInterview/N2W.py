

def numToWord(x):
    f1 = ["","One","Two","Three","Four","Five","Six","Seven", "Eight","Nine","Ten","eleven","Twelve","Thirteen","FourTeen","Fifteen","sixteen","SevenTeen",
          "Eighteen","ninteen"]
    f2 = ["","ten","Twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety","hundred"]
    wordNum = ''
    while x>0:
        if x >= 1000:
            wordNum = f1[x//1000] +" Thousand "
            x = x%1000
            continue
        elif x >= 100:
            wordNum = wordNum + f1[x//100] +" hundred "
            x = x % 100
            continue
        elif x >= 20:
            wordNum = wordNum +" "+ f2[x//10]
            x = x % 10
            continue
        else:
            wordNum = wordNum +" "+ f1[x]
            x = -1
            continue

    return wordNum

print("---------",numToWord(9876))