
class Solution(object):
    def groupAnagrams3(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dic = {}
        for str in strs:
            key = ''.join(sorted(str))
            print("key is  -------------",key)
            if key in dic:
                print("if YES key in dic: ------------", key)
                dic.get(key).append( str )
            else:
                dic[key] = [str]
                print("if NO key in dic: ------------", key)
        return dic.values()
    def mygroupanagram(self,strs):
        dict = {}
        for str in strs:
            key = ''.join(sorted(str))
            if key in dict:
                dict.get(key).append(str)
            else:
                dict[key]=[str]
        return dict.values()


if __name__ == '__main__':
    s = Solution()
    x =s.mygroupanagram(
        ["eat","tea","tan","ate","nat","bat"])
    print(x)