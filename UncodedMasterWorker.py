from __future__ import print_function
from time import sleep
import time
import sys
import random
import os
import numpy as np
from mpi4py import MPI
from numpy import linalg as LA
from util import *
from generate_data_helpers import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

expi = 1
scenario = 1
N = 156
mi = 6552
ni = 5000
alpha = 0.3
lrc = 0.001
rounds = 300
### overall data set dimensions
stragwrk = [0]*N
lclsize= mi/N
if rank == 0:	
	
	out2 = [None] * N
	for c1 in range(N):
		out2[c1]=np.zeros(ni)

	print ("At master",rank)
	
	timecaliter=np.zeros((rounds,1))
	erriter=np.zeros((rounds,1))
	timecalt=np.zeros((rounds,1))
	beta=np.random.normal(0,1,(ni,))
	
	# beta0 = np.random.uniform(-1,1,(ni,))
	
	beta0 = generate_random_label(ni)

	beta0 = comm.bcast(beta0, root=0)
	comm.Barrier()
	for i1 in range(rounds):
		lr=lrc/(i1+2)
		c7 = lr/mi

		g=np.zeros(ni)
		reqA = [None] * N
		reqC = [None] * N
		print (i1)
		comm.Barrier()
		
		bpsrt = time.time()
		for i2 in range(N):
			# print ('the special rank',i2+1)
			reqA[i2] = comm.Isend([beta, MPI.FLOAT], dest=i2+1)
		MPI.Request.Waitall(reqA)	

		for i2 in range(N):
			reqC[i2] = comm.Irecv([out2[i2], MPI.FLOAT], source=i2+1)


		for i3 in range(N):
			j = MPI.Request.Waitany(reqC)
			g=g+out2[j]

		betan=beta-c7*(g)-alpha*lr*beta
		beta=betan
		bpt = time.time()
		
		timecaliter[i1]=bpt-bpsrt
		timecalt[i1]=np.sum(timecaliter[0:(i1+1)])	
		erriter[i1]=pow((LA.norm(beta-beta0)/LA.norm(beta0)),2)
		print (timecalt[i1])
		print (erriter[i1])

	df=np.matrix(timecalt)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('UGC_time_%s_%s.txt'%(expi,scenario),df)	  
	
	df=np.matrix(erriter)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('UGC_error_%s.txt'%scenario,df)	  

	
else:
	rk=rank
	strag=stragwrk[rk-1];
	print ("At worker",rk)

	# Xi = load_data("UGC_X_"+ str(case)+"_"+str(rk) + ".dat")
	# yi = load_data("UGC_y_"+ str(case)+"_"+str(rk) + ".dat")
	beta0=np.zeros(ni)
    
	beta0 = comm.bcast(beta0, root=0)
	comm.Barrier()

	alpha = 10.5
	mu1 = np.multiply(alpha/ni, beta0)
	mu2 = np.multiply(-alpha/ni, beta0)
	Xi = generate_random_matrix_normal(mu1, mu2, lclsize,ni)
	yi=np.ndarray(lclsize)
	prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(Xi, beta0))))
	yi = np.random.binomial(1,prob_vals)
    
	beta=np.zeros(ni)

	for k in range(rounds):
		comm.Barrier()
		tst=time.time()
		reqBeta = None
		sC = None		

		reqBeta = comm.Irecv([beta, MPI.FLOAT], source=0)

		reqBeta.wait()

		#gradi=(np.transpose(Xi))*(Xi*beta-yi)
		# predy = Xi.dot(beta)
		prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(Xi, beta))))
		g = Xi.T.dot(prob_vals-yi)
		

		tr=time.time()-tst
		if strag==1:
			time.sleep(3*tr)
		
		sC = comm.Isend([g, MPI.FLOAT], dest=0)
		sC.wait()
