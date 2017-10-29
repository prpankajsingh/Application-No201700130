# Application-No201700130
Task 2 is to calculate weighted median.
To do that we must follow the following steps :
1. Calculate sigma, for this we mst claculate the magnitude of each vector obtained by the subtraction of S_i and S_target.
2. Calculate the weights of each training example.
3. Calculate the weighted Median.



Variables used :
N = total number of candidates
n = total number of good candidates
S[][] =  Nx36 matrices which contains the mouth space of N candidates in form a 36 dimensional vector
S_target[] = 36x1 vector which contains mouth shape of the target candidate.
w[] = nx1 vector which contains the weight of each candidate
alpha =  innitialised to 0.9(as mentioned in the paper)
sigma = variable to calculate weights.
