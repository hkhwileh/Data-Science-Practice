
nums = [2,7,11,15]
target = 13
size = len(nums)
for i in range(size):
    x = target - nums[i]
    for j in range(i+1,size):
        if x == nums[j]:
            print( [i,j])
