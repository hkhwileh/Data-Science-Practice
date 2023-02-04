class sol2:
    def fizzbizz(self,total):

        ans = []
        for num in range(total+1):

            devide_by_3 = (num % 3 == 0)
            devide_by_5 = (num % 5 == 0)
            
            if devide_by_3 and devide_by_5:
                ans.append("FizzBizz")
            elif devide_by_3:
                ans.append("Fizz")
            elif devide_by_5:
                ans.append("Bizz")
            else:
                ans.append(num)
        return ans


s = sol2()

print(s.fizzbizz(50))