﻿# Implmentation of K-bit weights re-ranking algorithm
import numpy as np
from tqdm import trange
import time
import math
import itertools
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

import pdb

class KBitWeights:
    def __init__(self, train_data, original_data, orig_eigens, k, max_iterations, logger, device):
        """
        Initializes a kBitWeights instance.

        Args:
            train_data: Training hash vectors.
            original_data: PCA processed original feature vectors of hasgh length
            Eigens: PCA eigne values of original data
            k: Number of non-zero elements above the diagonal in the upper-triangular matrix.
            max_iterations: Maximum number of iterations for training.
            logger: Write progress.
            device: CPU or GPU
        """
        self.train_data_hash = train_data
        self.hash_length = train_data.shape[1]
        self.original_data = original_data
        self.eigens = orig_eigens
        self.k = k
        self.max_iterations = max_iterations
        # self.W = torch.from_numpy(self.eigens).double().to(device)
        # self.W = np.zeros((self.hash_length, self.hash_length))
        # self.W = np.triu(np.ones((self.hash_length, self.hash_length)), k=0)
        # self.W = np.eye(self.hash_length)
        self.W = torch.ones((self.hash_length, self.hash_length), dtype=torch.float64, device=device)
        self.C = None
        self.log = logger
        self.device = device
        self.C_parallel = CreateCParallel(self.k, device_ids=[0, 1])

    def initialize_w(self):
        """
        Initialize W as an upper-triangular matrix with k elements above the diagonal.
        """
        for i in range(self.hash_length):
            for j in range(self.hash_length):
                if ((i == j) or ((j < (i + self.k)) and (j > i))):
                    self.W[i][j] = 1.0
                else:
                    self.W[i][j] = 0.0

    def generate_candidate_matrix(self, binary_vector):
        """
        Generate a candidate matrix C. Generate binary vectors by 
        inverting k bits in the input binary vector.

        Args:
            binary_vector (list): Input binary vector (list of 0s and 1s).
            k (int): Number of bits to invert.

        Returns:
            Matrx: A matrix of binary vectors with k bits inverted.
        """
        n = len(binary_vector)
        
        if self.k > n:
            raise ValueError("k cannot be greater than the length of the binary vector.")

        # Generate combinations of indices to flip
        flip_indices_combinations = list(itertools.combinations(range(n), self.k))
        
        # Initialize the matrix to store binary vectors
        matrix = np.zeros((len(flip_indices_combinations), n), dtype=int)
        
        for i, flip_indices in enumerate(flip_indices_combinations):
            # Copy the original binary vector
            new_vector = list(binary_vector)
            
            # Invert the bits at the selected indices
            for index in flip_indices:
                new_vector[index] = 1 - new_vector[index]
            
            # Assign the modified binary vector to a row in the matrix
            matrix[i] = new_vector
        
        return matrix
    
    def generate_candidate_matrix_gpu(self, binary_vector):
        """
        Generate a matrix by inverting k bits in the input binary vector using GPU acceleration.

        Args:
            binary_vector (list): Input binary vector (list of 0s and 1s).
            k (int): Number of bits to invert.

        Returns:
            torch.Tensor: Matrix with rows representing binary vectors with k bits inverted.
        """
        n = len(binary_vector)
        
        if self.k > n:
            raise ValueError("k cannot be greater than the length of the binary vector.")

        # Generate combinations of indices to flip
        flip_indices_combinations = list(itertools.combinations(range(n), self.k))
        
        # Initialize the matrix to store binary vectors on the GPU
        matrix_gpu = torch.zeros(len(flip_indices_combinations), n, dtype=torch.float64, device=self.device)
        
        # Copy the input binary vector to the GPU
        binary_vector_gpu = binary_vector.clone().detach()
        # binary_vector_gpu = torch.tensor(binary_vector, dtype=torch.int32, device=self.device)
        
        # Loop through each combination and invert bits using GPU
        for i, flip_indices in enumerate(flip_indices_combinations):
            # Copy the original binary vector to the GPU
            new_vector_gpu = binary_vector_gpu.clone()
            
            # Invert the bits at the selected indices on the GPU
            for index in flip_indices:
                new_vector_gpu[index] = 1 - new_vector_gpu[index]
            
            # Copy the modified binary vector from GPU to the matrix
            matrix_gpu[i] = new_vector_gpu
        
        # Copy the resulting matrix from GPU to CPU for further processing if needed
        matrix_cpu = matrix_gpu
        
        return matrix_cpu
                
    def train(self):
        """
        Train the model and return the updated W matrix.
        Returns:
            np.ndarray: The trained upper-triangular weight matrix W.
        """
        start_time = time.time()
        # pdb.set_trace()
        for iter in trange(self.max_iterations, desc="kbits algorithm training in progress.."):
            D = self.train_data_hash.double() @ self.W
            U,S,V = torch.svd(D.T)
            # print(U)
            # print(S)
            # print(V)
            self.W = torch.sum(V, dim=0)
            self.W = U * self.W
            # self.W += torch.var(self.original_data.double(), dim=0)
            # for row in self.train_data_hash:
            #     print(row)
                # self.C = self.generate_candidate_matrix(row)
                # self.C = self.generate_candidate_matrix_gpu(row)
                # shuffled_rows = torch.randperm(self.C.shape[0])
                # self.C = self.C[shuffled_rows].T
                # shuffled_cols = torch.randperm(self.C.shape[0])
                # self.C = self.C[shuffled_cols].T
                # Compute bit-wise variance
                # var = torch.var(self.C, dim=0)
                # self.W += var
                # self.C = self.C_parallel.generate_candidate_matrix_gpu(row)
                # print(f'The shape of C: {self.C.shape}')
                # D = self.W * self.C
                # D = np.dot(self.C, self.W)
                # Compute SVD, then update W
                # Compute the SVD
                # D = D.unsqueeze(dim=0)
                # U, S, Vt = torch.svd(D)
                # self.W = torch.sum(Vt, dim=0)
                # self.W = S
                # self.W = self.W / torch.sum(S)
                # self.W = Vt.squeeze()[:self.hash_length]
                # U: Left singular vectors
                # S: Singular values (a 1-D array of non-negative real numbers)
                # Vt: Right singular vectors (transpose of V)
                # You can reconstruct the original matrix using U, S, and Vt
                ##reconstructed_matrix = np.dot(U, np.dot(np.diag(S), Vt))
                # U = U[-self.hash_length:, :self.hash_length]
                # self.W = (S * Vt.T * self.W)
                # self.W = (U.T * Vt.T)
            # self.W = self.W / self.train_data_hash.shape[0]
            # for row in self.original_data:
            #     self.W += torch.var(self.original_data.double(), dim=0)
            # self.W = self.W / self.original_data.shape[0]
            # self.W += torch.var(self.train_data_hash.double(), dim=0)
        end_time = time.time()
        self.log.info('kBit algorithm training completed in {} secs.'.format(end_time - start_time))
        eigen = torch.tensor(self.eigens, dtype=torch.double, device=self.device)
        return torch.sum(self.W, dim=1) + eigen

class InvertBitsModel(nn.Module):
    def __init__(self, invert_bits_function):
        super(InvertBitsModel, self).__init__()
        self.invert_bits_function = invert_bits_function

    def forward(self, *args):
        return self.invert_bits_function(*args)

class CreateCParallel:
    def __init__(self, k, device_ids=None):
        # Initialize your class and specify the list of GPU device IDs to use
        self.device_ids = device_ids
        self.device = "cuda" if device_ids else "cpu"
        self.k = k

    def generate_candidate_matrix_gpu(self, binary_vector):
        n = len(binary_vector)

        if self.k > n:
            raise ValueError("k cannot be greater than the length of the binary vector.")

        # Generate combinations of indices to flip
        flip_indices_combinations = list(itertools.combinations(range(n), self.k))

        # Initialize the matrix on the GPU
        matrix_gpu = torch.zeros(len(flip_indices_combinations), n, dtype=torch.int32).to(self.device)

        # Copy the input binary vector to the GPU
        binary_vector_gpu = binary_vector.clone().detach()
        # binary_vector_gpu = torch.tensor(binary_vector, dtype=torch.int32).to(self.device)

        # Create a model wrapper for DataParallel
        parallel_model = InvertBitsModel(self._invert_bits)
        parallel_model = nn.DataParallel(parallel_model, device_ids=self.device_ids)  # Use nn.DataParallel

        # Call the model wrapper with arguments
        result_matrix_gpu = parallel_model(binary_vector_gpu, flip_indices_combinations)

        # Copy the resulting matrix from GPU to CPU for further processing if needed
        matrix_cpu = result_matrix_gpu.cpu()

        return matrix_cpu

    def _invert_bits(self, binary_vector_gpu, flip_indices_combinations):
        result_matrix = torch.zeros(len(flip_indices_combinations), len(binary_vector_gpu), dtype=torch.int32).to(self.device)

        for i, flip_indices in enumerate(flip_indices_combinations):
            new_vector_gpu = binary_vector_gpu.clone()

            for index in flip_indices:
                if index < len(new_vector_gpu):
                    new_vector_gpu[index] = 1 - new_vector_gpu[index]

            result_matrix[i] = new_vector_gpu

        return result_matrix







