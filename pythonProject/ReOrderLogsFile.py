#logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
logs = ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo","a2 act car"]

digits = []
letters = []
# divide logs into two parts, one is digit logs, the other is letter logs
for log in logs:
    if log.split()[1].isdigit():
        digits.append(log)
    else:
        letters.append(log)

print("before Sort",letters)
letters.sort()
print("after Sort",letters)
letters.sort()
print("after 2 Sort",letters)

