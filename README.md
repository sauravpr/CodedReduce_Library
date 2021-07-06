# CodedReduce_Library
To develop a full-fledged library to support straggler mitigation techniques inspired from Tree Gradient Coding

# Code Desription
UncodedMasterWorker.py: Implementation of distributed training using a master-worker setup. Master waits for results of all the clients
RingAllReduce.py: As name suggests, this is also distributed training, but the aggregation happens over ring using underlying API provided by mpi4py package
GradientCoding.py: This is the first work that proposed to use coding theoretic ideas to provide straggler mitigation https://arxiv.org/abs/1612.03301
TreeGradientCoding.py: This is the work I developed in collaboration with Amir, Prof. Ramtin, Prof. Salman https://web.ece.ucsb.edu/~ramtin/Tree_Gradient_Coding_final.pdf, https://arxiv.org/abs/1902.01981 
