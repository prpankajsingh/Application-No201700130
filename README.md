# Application-No201700130 
Task 1 is to find th 18 landmarks of mouth shape and then unroll it to a 36 dimensional vector and then apply Pricipal Component Analysis on the vector for dimensionality reduction.
To do the above task we must follow the following steps :
1. create a rectangular mask to detect the face from the picture. The function rect_to_bb does that.
2. We need to identify the important landmarks int face and store them. This is done inside the landmark function which produces a shape object that detects 68 landmark points for the face.(The mouth can be accessed through points [48, 68]. The right eyebrow through points [17, 22]. The left eyebrow through points [22, 27]. The right eye using [36, 42]. The left eye with [42, 48]. The nose using [27, 35].
And the jaw via [0, 17]). Since we just need 18 land marks for the mouth area so we convert the shape object to a numpy matrix using shape_to_np() function after which we store the mouth landmarks in a variable called lip.
3. Then we apply PCA for all the images. For each image wefind the 18 landmark points as stated above and then unroll it to a 36D vector which is stored in S[]. After which PCA for 20 component is applied and the value is stored in S_PCA. 


Task 2 is to calculate weighted median.
To do that we must follow the following steps :
1. Calculate sigma, for this we must calculate the magnitude of each vector obtained by the subtraction of S_i and S_target.Then we can calculate using equation 12.
2. Calculate the weights of each training example(equation 11).
3. Calculate the weighted Median.



Variables used :
N = total number of candidates
n = total number of good candidates
S[][] =  Nx36 matrices which contains the mouth space of N candidates in form a 36 dimensional vector
S_target[] = 36x1 vector which contains mouth shape of the target candidate.(assumed it is already provided)
w[] = nx1 vector which contains the weight of each candidate
alpha =  innitialised to 0.9(as mentioned in the paper)
sigma = variable to calculate weights.
I_i[] = an array of images loaded from the dataset
I_k = target image(assumed it is already provided).
median_img = contains the weighted median image(calculated via equation 9).
lip[] = contains 18 landmark  point for the mouth region
S_PCA = contain the 20 PCA coefficients.
