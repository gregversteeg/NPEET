#python2.7
#Testing the NPEET estimators

import entropy_estimators as ee
from math import log,pi
import numpy as np
import numpy.random as nr
import random
from numpy.linalg import det

#Some test cases to see usage and correctness

## Differential entropy estimator
print "For a uniform distribution with width alpha, the differential entropy is log_2 alpha, setting alpha = 2"
print "and using k=1,2,3,4,5"
print "result:", [ee.entropy([[2*random.random()] for i in range(1000)],k=j+1) for j in range(5)]

## CONDITIONAL MUTUAL INFORMATION
Ntry = [10,25,50,100,200] #,1000,2000] #Number of samples to use in estimate
nsamples = 100 #Number of times to est mutual information for CI
samplo = int(0.025*nsamples) #confidence intervals
samphi = int(0.975*nsamples)

print '\nGaussian random variables\n'
print 'Conditional Mutual Information'
d1 = [1,1,0]
d2 = [1,0,1]
d3 = [0,1,1]
mat = [d1,d2,d3]
tmat = np.transpose(mat)
diag = [[3,0,0],[0,1,0],[0,0,1]]
mean = np.array([0,0,0])
cov = np.dot(tmat,np.dot(diag,mat)) 
print 'covariance matrix'
print cov
trueent = -0.5*(3+log(8.*pi*pi*pi*det(cov))) 
trueent += -0.5*(1+log(2.*pi*cov[2][2])) #z sub
trueent += 0.5*(2+log(4.*pi*pi*det([[cov[0][0],cov[0][2]],[cov[2][0],cov[2][2]]] ))) #xz sub
trueent += 0.5*(2+log(4.*pi*pi*det([[cov[1][1],cov[1][2]],[cov[2][1],cov[2][2]]] ))) #yz sub
print 'true CMI(x:y|x)', trueent/log(2)

ent = []
err = []
for NN in Ntry:
  tempent = []
  for j in range(nsamples):
    points = nr.multivariate_normal(mean,cov,NN)
    x = [point[:1] for point in points] 
    y = [point[1:2] for point in points] 
    z = [point[2:] for point in points] 
    tempent.append(ee.cmi(x,y,z))
  tempent.sort()
  tempmean = np.mean(tempent)
  ent.append(tempmean)
  err.append((tempmean - tempent[samplo],tempent[samphi]-tempmean)) 

print 'samples used',Ntry
print 'estimated CMI',ent
print '95% conf int. (a,b) means (mean-a,mean+b)is interval\n',err

## MUTUAL INFORMATION

print 'Mutual Information'
trueent = 0.5*(1+log(2.*pi*cov[0][0])) #x sub
trueent += 0.5*(1+log(2.*pi*cov[1][1])) #y sub
trueent += -0.5*(2+log(4.*pi*pi*det([[cov[0][0],cov[0][1]],[cov[1][0],cov[1][1]]] ))) #xz sub
print 'true MI(x:y)', trueent/log(2)

ent = []
err = []
for NN in Ntry:
  tempent = []
  for j in range(nsamples):
    points = nr.multivariate_normal(mean,cov,NN)
    x = [point[:1] for point in points] 
    y = [point[1:2] for point in points] 
    tempent.append(ee.mi(x,y))
  tempent.sort()
  tempmean = np.mean(tempent)
  ent.append(tempmean)
  err.append((tempmean - tempent[samplo],tempent[samphi]-tempmean)) 

print 'samples used',Ntry
print 'estimated MI',ent
print '95% conf int.\n',err


print '\nIF you permute the indices of x, e.g., MI(X:Y) = 0'
#You can use shuffle_test method to just get mean, standard deviation
ent = []
err = []
for NN in Ntry:
  tempent = []
  for j in range(nsamples):
    points = nr.multivariate_normal(mean,cov,NN)
    x = [point[:1] for point in points] 
    y = [point[1:2] for point in points] 
    random.shuffle(y)
    tempent.append(ee.mi(x,y))
  tempent.sort()
  tempmean = np.mean(tempent)
  ent.append(tempmean)
  err.append((tempmean - tempent[samplo],tempent[samphi]-tempmean)) 

print 'samples used',Ntry
print 'estimated MI',ent
print '95% conf int.\n',err

## DISCRETE ESTIMATORS

print "\n\nTest of the discrete entropy estimators\n"
print "For z = y xor x, w/x,y uniform random binary, we should get H(x)=H(y)=H(z) = 1, H(x:y) etc = 0, H(x:y|z) = 1"
x = [0,0,0,0,1,1,1,1]
y = [0,1,0,1,0,1,0,1]
z = [0,1,0,1,1,0,1,0]
print "H(x),H(y),H(z)",ee.entropyd(x),ee.entropyd(y),ee.entropyd(z)
print "H(x:y),etc",ee.midd(x,y),ee.midd(z,y),ee.midd(x,z)
print "H(x:y|z),etc",ee.cmidd(x,y,z),ee.cmidd(z,y,x),ee.cmidd(x,z,y)


## KL Div estimator
print "\n\nKl divergence estimator (not symmetric, not required to have same num samples in each sample set"
print "should be 0 for same distribution"
sample1 = [[2*random.random()] for i in range(200)]
sample2 = [[2*random.random()] for i in range(300)]
print 'result:',ee.kldiv(sample1,sample2)
print "should be infinite for totally disjoint distributions (but this estimator has an upper bound like log(dist) between disjoint prob. masses)"
sample2 = [[3+2*random.random()] for i in range(300)]
print 'result:',ee.kldiv(sample1,sample2)
