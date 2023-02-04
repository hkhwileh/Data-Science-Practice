# creating a list containing elements
# belonging to different data types
# defining list
sample_list = [1,"Yash",['a','e']]

array = [1,2,3,4]
aExtend = [2,3,4]
'''
print(sample_list
print("----------",type(array))
print("----------",type(sample_list))

i = array.pop()
print(i)

array.append(22)'''
# insert & append & index
array.insert(22,44)
#print(array)
#print(array.index(44))

# extend

array.extend(aExtend)
#print(array)

#remove with value

array.remove(44)
#print(array)

# remove the last with pop
i = array.pop(0)
#print(array)

# slice print

list_num = [1,2,3,4,5,6,7,8,9,10]


'''List comprehensions are used for creating new lists from other iterables like tuples, strings, arrays, lists, etc.
A list comprehension consists of brackets containing the expression, which is executed for each 
element along with the for loop to iterate over each element'''

comrehension = [i*2 for i in range(1,22) if i%2==0 ]
print(comrehension)

print(comrehension.count(40))