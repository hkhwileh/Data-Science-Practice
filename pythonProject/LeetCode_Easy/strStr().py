haystack = "hello"
needle = "ll"
print("len(haystack)",len(haystack))
print("len(needle)",len(needle))

print("len(haystack) - len(needle) + 1",len(haystack) - len(needle) + 1)
for i in range(len(haystack)):
    if haystack[i+len(needle)]==needle:
        print(i)
