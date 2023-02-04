
nums = [1,3,5,6]
target = 5
left , right = 0 ,len(nums)-1

while left <= right:
    mid = (left+right)//2
    if nums[mid]==target:
        print( mid)
    elif target <nums[mid]:
        right = mid - 1
    else:
        left = mid + 1
print( mid)