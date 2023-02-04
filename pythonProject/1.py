
s = "00110011"

if s.count("0") == 0 or s.count("1") == 0:
    print (0)
s = s.replace("01", "0 1")
s = s.replace("10", "1 0")
s = s.split(" ")
count = 0
for i in range(len(s) - 1):
    print("    print(len(s[i]))",len(s[i]))
    print("    print(len(s[i+1]))",len(s[i+1]))

    count = count + min(len(s[i]), len(s[i + 1]))
    print("count",count)
print( count)