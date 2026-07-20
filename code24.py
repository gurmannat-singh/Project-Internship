import numpy as np 
a = np.array([1,2,3])
print (a)

#2-d array
ab = np.array([[1,2,3],[2,3,4]])
print (ab)

#3-d array
b = np.array([[[1,2],[2,3]],[[4,5],[2,3]],[[1,2],[2,3]]])
print (b)
print(b.shape)

a=np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,19],[11,12]],[[13,14],[15,16]]])
print(a)
print(a.shape)
print(a.ndim)
print(a.size)
print(a.dtype)
print(a.astype) # already have an array then astype is used to change datatype
#shape (rows and columns) and (no of 2d array ,rows and columns)
#ndim gives no pof dimensions of array 

zeroes_array = np.zeros((3,3))
print(zeroes_array)

zeroes_array = np.zeros((3,3), dtype= int)
print(zeroes_array)

first_array = np.ones((3,3))
print(first_array)
first_array = np.ones((3,3),dtype = int)
print(first_array)

identity_array = np.identity(5)
print(identity_array)

#random
random_array = np.random.random((2,3))
print(random_array)

#arrange
arrange_array = np.arange(1,11,2)
print(arrange_array)

#full 
full_array = np.full((3,3),4)
print(full_array)

#astype
z = np.array([2,3,4,5,6,7,8])
m= z.astype("float64")
print(m)

q = np.linspace(1,11,6,endpoint=False, retstep = True)
print(q)


random_array = np.random.randint((2,3))
print(random_array)