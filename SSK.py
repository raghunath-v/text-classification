#author ='Jihane BOUHAMMADA'


import numpy as np
import sys


# Parameters :
	# lambda_decay is the decay factor
	# small k is the length of subsequences

# computing K'
def kPrime(k, s, t, lambda_decay=0.5):
	if k == 0:
		return 1
	if min(len(s), len(t)) < k:
		return 0
	else:
		sum_ = 0
		for j in range(1, len(t)):
			if t[j] == s[-1]:
				sum_ += kPrime(k-1, s[:-1], t[:j]) * (lambda_decay ** (len(t) - (j+1) + 2))
		result = lambda_decay * kPrime(k, s[:-1], t) + sum_
	return result


# computing Kn
def kKernel(k, s, t, lambda_decay=0.5):
	if k == 0:
		return 1
	if min(len(s), len(t)) < k:
		return 0
	else:
		sum_ = 0
		for j in range(1, len(t)):
			if t[j] == s[-1]:
				sum_ += kPrime(k-1, s[:-1], t[:j])
		result = kKernel(k, s[:-1], t) + lambda_decay ** 2 * sum_
	return result


# computing SSK
def gramMatrixElements(k, s, t, ssValue, ttValue):
	if s == t:
		return 1
	else:
		try:
			return kKernel(k, s, t) / (ssValue * ttValue) ** 0.5
		except ZeroDivisionError:
			print("Error: The maximal susbsequences length is less or equal to documents' minimal length.")
			sys.exit(2)



# computing SSK
def stringKernel(s, t):
	S = len(s)
	T = len(t)
	
	gramMatrix = np.zeros((S, T), dtype=np.float32)
	simValue = {}
	if s == t:
		for i in range(S):
			simValue[i] = kKernel(k, s[i], t[i])
		for i in range(S):
			for j in range(i, T):
				gramMatrix[i, j] = gramMatrixElements(k, s, t, simValue[i], simValue[j])
				#gramMatrix[i][j] = gramMatrix[j][i]
				gramMatrix[i, j] = gramMatrix[j, i]
	elif S == T:
		simValue[1] = {}
		simValue[2] = {}
		for i in range(S):
			simValue[1][i] = kKernel(k, s[i], s[i])
		for j in range(T):
			simValue[2][j] = kKernel(k, t[j], t[j])
		for i in range(S):
			for j in range(i, T):
				gramMatrix[i, j] = gramMatrixElements(k, s[i], t[j], simValue[1][i], simValue[2][j])
				gramMatrix[i, j] = gramMatrix[j, i]
	else:
		simValue[1] = {}
		simValue[2] = {}
		m = min(S, T)
		for i in range(S):
			simValue[1][i] = kKernel(k, s[i], s[i])
		for j in range(T):
			simValue[2][j] = kKernel(k, t[j], t[j])
		for i in range(m):
			for j in range(i, m):
				gramMatrix[i, j] = gramMatrixElements(k, s[i], t[j], simValue[1][i], simValue[2][j])
				gramMatrix[i, j] = gramMatrix[j, i]

		if S > T:
			for i in range(m, T):
				for j in range(T):
					gramMatrix(k, s[i], t[j], simValue[1][i], simValue[2][j])
		else:
			for i in range(S):
				for j in range(m, T):
					gramMatrix(s[i], t[j], simValue[1][i], simValue[2][j])
	return gramMatrix




k = 2
s = ['car', 'cat', 'fat']
t = ['bar', 'bat', 'mat']
print(kKernel(k, s, t))
print(stringKernel(s, t))