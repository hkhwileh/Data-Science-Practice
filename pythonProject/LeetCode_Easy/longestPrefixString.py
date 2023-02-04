


def longestCommonPrefix(strs):
    if not strs:
        return ''
    i = 0
    for x in zip(*strs):

        if len(set(x)) > 1:
            return strs[0][:i]
        i += 1
    return strs[0][:i] if strs else ''


print(longestCommonPrefix(["flower","flow","flight"]))