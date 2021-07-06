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

N=156
S=int(sys.argv[1])
mi=6552
ni=5000
expi = 1
scenario = 3

# N=90
# S=int(sys.argv[1])
# mi=6300
# ni=5800
# expi = 2
# scenario = 2
alpha = 0.3
lrc = 0.001
rounds = 300
stragwrk = [0]*N
lclsize= (S+1)*mi/N

if rank == 0:   
	B = np.zeros((N,N))

	B = getB(N,S)
    # print("B loading complete")
    # B = comm.bcast(B, root=0)
	msgBuffers = np.array([np.zeros(ni) for i in range(N)])
	g=np.zeros(ni)
	A_row = np.zeros((1,N))
	cnt_completed = 0
	completed_workers = np.ndarray(N,dtype=bool)
	beta=np.ones(ni)
	# beta0 = load_data("beta_"+str(case)+"_0.dat")
	timecaliter=np.zeros((rounds,1))
	erriter=np.zeros((rounds,1))
	timecalt=np.zeros((rounds,1))
	
	beta0 = generate_random_label(ni)
	beta0 = comm.bcast(beta0, root=0)
	comm.Barrier()
	for i1 in range(rounds):
		lr=lrc/(i1+2)
		c7 = lr/mi

		print(i1)
		comm.Barrier()
		reqA = [None] * N
		reqC = [None] * N
		A_row[:] = 0
		completed_workers[:]=False
		cnt_completed = 0
		bpsrt = time.time()
		for i2 in range(N):
			reqA[i2] = comm.Isend([beta, MPI.FLOAT], dest=i2+1,tag=1)
		MPI.Request.Waitall(reqA)
		for i2 in range(N):
			reqC[i2] = comm.Irecv([msgBuffers[i2], MPI.FLOAT], source=i2+1)
		for i3 in range((N-S)):
			j = MPI.Request.Waitany(reqC)
			completed_workers[j] = True
		completed_ind_set = [l for l in range(N) if completed_workers[l]]
		A_row[0,completed_ind_set] = np.linalg.lstsq(B[completed_ind_set,:].T,np.ones(N))[0]
		g = np.squeeze(np.dot(A_row, msgBuffers))
		betan=beta-c7*(g)-alpha*lr*beta
		beta=betan
		bpt = time.time()
		
		timecaliter[i1]=bpt-bpsrt
		timecalt[i1]=np.sum(timecaliter[0:(i1+1)])  
		erriter[i1]=pow((LA.norm(beta-beta0)/LA.norm(beta0)),2)
		print (timecalt[i1])
		print (erriter[i1])
	print(S)
	print(timecalt[-1])    
	df=np.matrix(timecalt)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('MWCGC_time_%s_%s_%s.txt'%(expi,scenario,S),df)    

	df=np.matrix(erriter)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('MWCGC_error_%s_%s_%s.txt'%(expi,scenario,S),df)    
	comm.Barrier()
else:
	rk=rank
	i = rank
	strag=stragwrk[rk-1]
	# print ("At worker",rk)
	
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
		# print ("rounds started",k)
		tst = time.time()
		reqBeta = None
		sC = None
		reqBeta = comm.Irecv([beta, MPI.FLOAT], source=0,tag=1)
		reqBeta.wait()

		prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(Xi, beta))))

		g = Xi.T.dot(np.multiply(prob_vals-yi,np.ones(lclsize)))

		tr = time.time()-tst
		if strag==1:
			time.sleep(3*tr)
		sC = comm.Isend([g, MPI.FLOAT], dest=0)
		sC.wait()
	comm.Barrier()
