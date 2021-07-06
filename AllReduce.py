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
size = comm.Get_size()

expi = 1
scenario = 2
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
	print ("At master",rank)
	masterComm = comm.Split( 0, rank )
	timecaliter=np.zeros((rounds,1))
	erriter=np.zeros((rounds,1))
	timecalt=np.zeros((rounds,1))
	beta=np.random.normal(0,1,(ni,))
	
	# beta0 = np.random.uniform(-1,1,(ni,))
	
	beta0 = generate_random_label(ni)

	beta0 = comm.bcast(beta0, root=0)
	comm.Barrier()
	
	for i1 in range(rounds):
		

		print (i1)
		comm.Barrier()
		recvmsg1 = comm.gather(0, root=0)
		# print rank, len(recvmsg1)
		timecaliter[i1]=max(recvmsg1[1:])
		timecalt[i1]=np.sum(timecaliter[0:(i1+1)])
		reqF=comm.Irecv([beta, MPI.FLOAT], source=1)
		reqF.wait()
		erriter[i1]=pow((LA.norm(beta-beta0)/LA.norm(beta0)),2)
		print (timecalt[i1])
		print (erriter[i1])

	df=np.matrix(timecalt)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('AGC_time_%s_%s.txt'%(expi,scenario),df)	  
	
	df=np.matrix(erriter)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('AGC_error_%s_%s.txt'%(expi,scenario),df)	  

	
else:
	rk=rank
	workerComm = comm.Split( 1, rank );

	strag=stragwrk[rk-1];
	print ("At worker",rk)
	beta0=np.zeros(ni)
    
	beta0 = comm.bcast(beta0, root=0)
	comm.Barrier()

	# Xi = load_data("UGC_X_"+ str(case)+"_"+str(rk) + ".dat")
	# yi = load_data("UGC_y_"+ str(case)+"_"+str(rk) + ".dat")
	alpha = 10.5
	mu1 = np.multiply(alpha/ni, beta0)
	mu2 = np.multiply(-alpha/ni, beta0)
	Xi = generate_random_matrix_normal(mu1, mu2, lclsize,ni)
	yi=np.ndarray(lclsize)
	prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(Xi, beta0))))
	yi = np.random.binomial(1,prob_vals)
    
	beta=np.ones(ni)
	alpha=0.3
	
	for k in range(rounds):
		lr=lrc/(k+2)
		c7 = lr/mi


		grad=np.zeros(ni)
		comm.Barrier()
		workerComm.Barrier()
		tst=time.time()
		
		# if rank ==1:
		# 	print(Xi.shape)
		# 	print(yi.shape)
		# 	print(beta.shape)
		prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(Xi, beta))))
		gradi = Xi.T.dot(prob_vals-yi)
		
		
		tr=time.time()-tst
		if strag==1:
			time.sleep(3*tr)
		
		workerComm.Allreduce(gradi,grad,op=MPI.SUM)
		betan=beta-c7*(grad)-alpha*lr*beta
		beta=betan

		bpt = time.time()


		tlocal=bpt-tst

		recvmsg1 = comm.gather(tlocal, root=0)


		if rk==1:
			sC = comm.Isend([beta, MPI.FLOAT], dest=0)
			sC.wait()
			if k==99:
				print("done at the workers")
			# print rk,recvmsg1
		
