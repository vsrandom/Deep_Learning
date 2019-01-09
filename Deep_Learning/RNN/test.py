# reshape 2D array
from numpy import array
# list of data
data = [1,2,3,4,5]
#print(data)
data=array(data)
print(data.shape)
print(20*'$')
#print(data.shape)
data = data.reshape((-1,data.shape[0]))
print(data.shape)
print(data)
