list1 = []
list2 = []
list1.append((1,1))
list1.append((1,2))
list2.append((1,1))
list3 = set(list1) - set(list2)
print(list3)