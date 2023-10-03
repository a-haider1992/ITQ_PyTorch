# Implmentation of K-bit weights re-ranking algorithm
import numpy as np
from tqdm import trange
import time
import math
import itertools
import torch

class KBitWeights:
    def __init__(self, train_data, k, max_iterations, logger, device):
        """
        Initializes a kBitWeights instance.

        Args:
            train_data: Training hash vectors.
            k: Number of non-zero elements above the diagonal in the upper-triangular matrix.
            max_iterations: Maximum number of iterations for training.
            logger: Write progress.
            device: CPU or GPU
        """
        self.train_data_hash = train_data
        self.hash_length = train_data.shape[1]
        self.k = k
        self.max_iterations = max_iterations
        self.W = np.zeros((self.hash_length, self.hash_length))
        self.C = None
        self.log = logger
        self.device = device

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
        matrix_gpu = torch.zeros(len(flip_indices_combinations), n, dtype=torch.int32, device=self.device)
        
        # Copy the input binary vector to the GPU
        binary_vector_gpu = torch.tensor(binary_vector, dtype=torch.int32, device=self.device)
        
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
        matrix_cpu = matrix_gpu.cpu()
        
        return matrix_cpu
                
    def train(self):
        """
        Train the model and return the updated W matrix.
        Returns:
            np.ndarray: The trained upper-triangular weight matrix W.
        """
        start_time = time.time()
        for iter in trange(self.max_iterations, desc="kbits algorithm training in progress.."):
            for row in self.train_data_hash:
                # self.C = self.generate_candidate_matrix(row)
                self.C = self.generate_candidate_matrix_gpu(row)
                print(f'The shape of C: {self.C.shape}')
                # D = np.dot(self.C, self.W)
                # Compute SVD, then update W
        end_time = time.time()
        self.log.info('kBit algorithm training completed in {} secs.'.format(end_time - start_time))
        return self.W

