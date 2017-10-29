import math
def vectorMag(S,S_target,n):
	sum=0,sqSum=0
	for i in range(n):
		sqSum = 0
		for j in range(36):
			sqSum+= (S[i][j]-S_target[j])^2
		sum+=sqSum
	return sum
 
def Calc_sigma(S,S_target):
	sigma = (vectorMag(S,S_target,n) - vectorMag(S,S_target,N))/(2*(-0.10536))
	return sigma
 
def Calc_weight(S, S_target):
	sqSum=0
	sigma = Calc_sigma(S,S_target);
	for i in range(n):
		sqSum = 0
		for j in range(36):
			sqSum+= (S[i][j]-S_target[j])^2
			p= sqSum/(2*sigma*sigma)
        w[i] = e^-p
 
alpha=0.9
w=[0]* N
