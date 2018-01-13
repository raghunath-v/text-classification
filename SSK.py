#author ='Jihane BOUHAMMADA'


import numpy as np

# Parameters :
	# lambda_decay is the decay factor
	# small k is the length of subsequences

# computing K'
def kPrime(k, s, t, lambda_decay=0.5):

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
		return kKernel(k, s, t) / ((ssValue * ttValue) ** 0.5)


# computing SSK
def stringKernel(s, t):

	


k = 2
s = 'cat'
t = 'car'
print(kKernel(k, s, t))




