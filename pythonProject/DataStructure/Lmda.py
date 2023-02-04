def interSection(arr1, arr2):
    # filter(lambda x: x in arr1, arr2)  -->
    # filter element x from list arr2 where x
    # also lies in arr1
    result = list(filter(lambda x: x in arr1, arr2))
    print("Intersection : ", result)


# Driver program
'''if __name__ == "__main__":
    arr1 = [1, 3, 4, 5, 7]
    arr2 = [2, 3, 5, 6]
    interSection(arr1, arr2)'''


# Array

import array as arr
a = arr.array('str',[1,2,3,4,5,6])

for i in a:
    print(i)
