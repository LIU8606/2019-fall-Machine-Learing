import numpy as np
import matplotlib.pyplot as plt
import math

def LU(A):
	
	n = len(A) 

	L = [[0.0 for i in range(n)] for i in range(n)]
	for i in range(0,n):
		L[i][i] = 1.0
	L = np.array(L)

	U = np.copy(A)

	for i in range(0,n-1):
		for k in range(i+1,n):
			if float(U[i][i])!=0:
				c = U[k][i]/float(U[i][i])
				L[k][i] = c 
				for j in range(i, n):
					U[k][j] -= c*U[i][j]

	return L,U


def inverse(A):

	n = len(A)

	b = np.identity(n)

	L,U = LU(A)
	

	y = [[0 for _ in range(n)] for _ in range(n)]
	for row in range(n):
		for i in range(0, n):
			tmp = b[row][i]
			for j in range(0,n):
				tmp -= y[row][j]*L[i][j]

			y[row][i] = tmp/L[i,i]

	x = [[0 for _ in range(n)] for _ in range(n)]

	for row in range(n):
		for i in range(n-1,-1,-1):
			tmp = y[row][i]
			for j in range(n-1,-1,-1):
				tmp -= x[row][j]*U[i][j]
			x[row][i] = tmp/U[i,i]

	return np.transpose(x)

def calu_point(X,Y):
	l_y = []
	for i in range(len(X)):
		tmp = 0
		for j in range(len(Y)):
			tmp += (X[i]**(n-1-j))*Y[j]
		l_y.append(tmp)

	return l_y



filename = input("input filename:")
n = int(input("input n:"))
lamda = int(input("input lamda:"))

###read file###
f = open(filename,'r')

X =[]
Y=[]

for line in f:
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

I = np.array([[lamda*(i ==j) for i in range(n)] for j in range(n)])
b = np.array([Y[i] for i in range(len(Y))])
A = np.array([[(X[i])**(n-1-p) for p in range(n) ] for i in range(len(X))])
                                                                                                                                                                                                                    
ATA = np.dot(np.transpose(A),A)

###LSE###
ATA_I = ATA + I
ATA_I_inverse = inverse(ATA_I)

ATAAT = np.dot(ATA_I_inverse,np.transpose(A))
ATAATB = np.dot(ATAAT,b)

#print(ATAATB)
#R = np.linalg.lstsq(A, b)
#print(R[0])

X_draw = np.copy(X)
X.sort()

l_y_LU = calu_point(X,ATAATB)

###Newton###
x_0 = [ 0 for _ in range(n)]
x_1 = [ 0 for _ in range(n)]

DF = 2* np.dot(ATA,x_0) - 2*np.dot(np.transpose(A),b)
HF = 2*ATA

x_1 = x_0 - np.dot(inverse(HF),DF)
#x_1 = x_0 - np.dot(np.linalg.inv(HF),DF)

l_y_NewTon = calu_point(X,x_1)

###print####
print("LSE")
print("Fitting line: ",end='')
for i in range(n):
	power = n-1-i
	print(ATAATB[i],end='')
	if i != n-1:
		if ATAATB[i+1]>=0:
			print("X^"+ str(power)+" + ",end='')
		else:
			print("X^"+ str(power)+"   ",end='')
print()
print("Total error:"+str(sum((np.array(l_y_LU)-b)**2)))

print()

print("Newton's Method:")
print("Fitting line: ",end='')
for i in range(n):
	power = n-1-i
	print(x_1[i],end='')
	if i != n-1:
		if x_1[i+1]>=0:
			print("X^"+ str(power)+" + ",end='')
		else:
			print("X^"+ str(power)+"   ",end='')
print()
print("Total error:"+str(sum((np.array(l_y_NewTon)-b)**2)))

###draw###
plt.subplot(211)
plt.plot(X, l_y_LU, linestyle='solid')
plt.plot(X_draw, Y, 'ro')

plt.subplot(212)
plt.plot(X, l_y_NewTon, linestyle='solid')
plt.plot(X_draw, Y, 'ro')
plt.show()






