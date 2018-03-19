import random

def test2(T):
	result = 1
	maxT = T[0]
	temp = T[0]
	for i in range(1, len(T)):
		if maxT < T[i]:
			if temp < T[i]:
				temp = T[i]
		else:
			maxT = temp
			result = i+1
	print(result)

def test3(T):
	i = 1
	while i < len(T)-1:
	    if max(T[0:i])<min(T[i:]):
	        print(i)
	        break
	    else:
	        i = i+1

for i in range(100):
	T = []
	for i in range(5):
		T.append(random.choice(range(-5,5)))
	print(T)
	test2(T)
	test3(T)


