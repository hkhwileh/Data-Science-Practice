

def reorderLogFiles(logs):


    digits = []
    letters = []
        # divide logs into two parts, one is digit logs, the other is letter logs
    for log in logs:
        if log.split()[1].isdigit():
            digits.append(log)
            print(log.split()[1])
        else:
            letters.append(log)

    letters.sort(key=lambda x: x.split()[1])  # when suffix is tie, sort by identifier
    letters.sort(key=lambda x: x.split()[1:])  # sort by suffix

    result = letters + digits  # put digit logs after letter logs
    print("result-------------",result)

logs = ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
reorderLogFiles(logs)

