def merge_sort(arr):
    new_arr = []
    if len(arr)>1 :
        left_arr = arr[:len(arr)//2]
        right_arr = arr[len(arr)//2:]

        #recursion
        merge_sort(left_arr)
        merge_sort(right_arr)

        #merge
        i = 0
        j = 0
        k = 0

        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] < right_arr[j]:
                new_arr[k]= left_arr[i]
                i +=1
            else:
                new_arr[k] = right_arr[j]
                j += 1
            k+=1
        while i < len(left_arr) :
            new_arr[k] = left_arr[i]
            i += 1
            k += 1
        while j < len(right_arr) :
            new_arr[k] = right_arr[j]
            j += 1
            k += 1
    return new_arr


arr_test = [2,3,5,1,7,4,4,4,2,6,0]


print(merge_sort(arr_test))