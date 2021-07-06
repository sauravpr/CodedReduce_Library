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
######################################################Need to modify the y_current_mod for the logistic regression...
###########################but does not affect the result###################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

expi = 1
scenario = 4

N=156
L=2
n=12
s=int(sys.argv[1])

mi=int(sys.argv[2])
ni=5000
alpha = 0.3
lrc = 0.001
rounds = 300
### overall data set dimensions
stragwrk = [0]*N
c5=0
for i1 in range(L):
    c5=c5+pow(n,i1+1)
c4=c5-pow(n,L)
layerL=set([x + 1 for x in range(c4,c5,1)])

c12=0
for i1 in range(L):
	c12=c12+pow(1.0*n/(s+1),i1+1)

lclsize=int(mi/c12)
B = np.zeros((n,n))
if rank == 0:   
	B=getB(n,s)
    # print("B loading complete")
	B = comm.bcast(B, root=0)
	comm.Barrier()
	msgBuffers = np.array([np.zeros(ni) for i in range(n)])
	g=np.zeros(ni)
	A_row = np.zeros((1,n))
	cnt_completed = 0
	completed_workers = np.ndarray(n,dtype=bool)

	beta=np.random.normal(0,1,(ni,))
	
	# beta0 = np.random.uniform(-1,1,(ni,))
	
	beta0 = generate_random_label(ni)

	beta0 = comm.bcast(beta0, root=0)
	comm.Barrier()
	
	timecaliter=np.zeros((rounds,1))
	erriter=np.zeros((rounds,1))
	timecalt=np.zeros((rounds,1))
	for i1 in range(rounds):
		print(i1+1)
		lr=lrc/(i1+2)
		c7 = lr/mi

		# print(i1)
		comm.Barrier()
		reqA = [None] * n
		reqC = [None] * n
		A_row[:] = 0
		completed_workers[:]=False
		cnt_completed = 0
		bpsrt = time.time()
		for i2 in range(n):
			reqA[i2] = comm.Isend([beta, MPI.FLOAT], dest=i2+1,tag=1)
		MPI.Request.Waitall(reqA)
		for i2 in range(n):
			reqC[i2] = comm.Irecv([msgBuffers[i2], MPI.FLOAT], source=i2+1)
		for i3 in range((n-s)):
			j = MPI.Request.Waitany(reqC)
			completed_workers[j] = True
		completed_ind_set = [l for l in range(n) if completed_workers[l]]
		A_row[0,completed_ind_set] = np.linalg.lstsq(B[completed_ind_set,:].T,np.ones(n))[0]
		g = np.squeeze(np.dot(A_row, msgBuffers))
		betan=beta-c7*(g)-alpha*lr*beta
		beta=betan
		bpt = time.time()
		timecaliter[i1]=bpt-bpsrt
		timecalt[i1]=np.sum(timecaliter[0:(i1+1)])
		print (timecalt[i1])  
		erriter[i1]=pow((LA.norm(beta-beta0)/LA.norm(beta0)),2)
        
	print (s)
	print(timecalt[-1])         
	df=np.matrix(timecalt)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('TCGC_time_%s_%s_%s.txt'%(expi,scenario,s),df)    

	df=np.matrix(erriter)
	#np.savetxt('time_uncoded_master.txt',df)
	np.savetxt('TCGC_error_%s_%s_%s.txt'%(expi,scenario,s),df)    

else:
	rk=rank
	i = rank
	B = comm.bcast(B, root=0)
	comm.Barrier()
	beta0=np.zeros(ni)
    
	beta0 = comm.bcast(beta0, root=0)
	comm.Barrier()

	prank=(rk-1)/n # parent rank
	strag=stragwrk[rk-1]
	# print ("At worker",rk)
	
    # print("B received at workers")
	g = np.zeros(ni)
	alpha1 = 10.5
	mu1 = np.multiply(alpha1/ni, beta0)
	mu2 = np.multiply(-alpha1/ni, beta0)
	Xi = generate_random_matrix_normal(mu1, mu2, lclsize,ni)
	yi=np.ndarray(lclsize)
	prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(Xi, beta0))))
	yi = np.random.binomial(1,prob_vals)
    
	beta=np.zeros(ni)

	y_current_mod = np.random.normal(0,1,(lclsize,))


    # Xi = load_data("TCGC_X_"+ str(case)+"_"+str(i) + ".dat")
    # yi = load_data("TCGC_y_"+ str(case)+"_"+str(i) + ".dat")
    # y_current_mod = load_data("TCGC_curr_"+ str(case)+"_"+str(i) + ".dat")
	beta=np.zeros(ni)
	if rk in layerL: # need to wait for (n-s)
		for k in range(rounds):
			comm.Barrier()
            # print ("rounds started",k)
			tst = time.time()
			reqBeta = None
			sC = None
			reqBeta = comm.Irecv([beta, MPI.FLOAT], source=prank,tag=1)
			reqBeta.wait()

			prob_vals = np.divide(1.,np.add(1,np.exp(-np.dot(Xi, beta))))
			g = Xi.T.dot(np.multiply(prob_vals-yi,np.ones(lclsize)))

			# simulating the product from B coefficients
			tr = time.time()-tst
			if strag==1:
				time.sleep(3*tr)
			sC = comm.Isend([g, MPI.FLOAT], dest=prank)
			sC.wait()

            # print("beta received at each worker")
            
	else:
		msgBuffers = np.array([np.zeros(ni) for i in range(n)])
		A_row = np.zeros((1,n))
		cnt_completed = 0
		completed_workers = np.ndarray(n,dtype=bool)
		temp1 = list(([n*rk + x + 1 for x in range(n)]))
		for k in range(rounds):
			comm.Barrier()
			# print("rounds",k)
			A_row[:] = 0
			completed_workers[:]=False
			cnt_completed = 0
			reqA = [None] * n
			reqB = [None] * n
			reqBeta = None
			sC = None
			tst=time.time()
			reqBeta = comm.Irecv([beta, MPI.FLOAT], source=prank,tag=1)
			reqBeta.wait()
			for i3 in range(n):
				reqA[i3] = comm.Isend([beta, MPI.FLOAT], dest=temp1[i3],tag=1)
			MPI.Request.Waitall(reqA)

			gradi=Xi.T.dot(np.multiply(prob_vals-yi,np.ones(lclsize)))
			tr=time.time()-tst
			if strag==1:
				time.sleep(3*tr)
			for i2 in range(n):
				reqB[i2] = comm.Irecv([msgBuffers[i2], MPI.FLOAT], source=temp1[i2])
			for i3 in range((n-s)):
				j = MPI.Request.Waitany(reqB)
				completed_workers[j] = True
			completed_ind_set = [l for l in range(n) if completed_workers[l]]
			A_row[0,completed_ind_set] = np.linalg.lstsq(B[completed_ind_set,:].T,np.ones(n))[0]
			g = np.squeeze(np.dot(A_row, msgBuffers))+gradi

			sC = comm.Isend([g, MPI.FLOAT], dest=prank)
			sC.wait()