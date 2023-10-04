import torch
import numpy as np

from sklearn.decomposition import PCA
from utils.evaluate import mean_average_precision, pr_curve
from kbit_weight import KBitWeights

import pdb

def train(
    train_data,
    query_data,
    query_targets,
    retrieval_data,
    retrieval_targets,
    code_length,
    max_iter,
    device,
    topk,
    k,
    logger
    ):
    """
    Training model.

    Args
        train_data(torch.Tensor): Training data.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): Retrieval targets.
        code_length(int): Hash code length.
        max_iter(int): Number of iterations.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.
        k(int): k = [1, code_length/2] significant bits of generated hash codes.
        logger: Logging progress.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Initialization
    query_data, query_targets, retrieval_data, retrieval_targets = query_data.to(device), query_targets.to(device), retrieval_data.to(device), retrieval_targets.to(device)
    R = torch.randn(code_length, code_length).to(device)
    [U, _, _] = torch.svd(R)
    R = U[:, :code_length]

    # PCA
    pca = PCA(n_components=code_length)
    V = torch.from_numpy(pca.fit_transform(train_data.numpy())).to(device)

    # Training
    for i in range(max_iter):
        V_tilde = V @ R
        B = V_tilde.sign()
        [U, _, VT] = torch.svd(B.t() @ V)
        R = (VT.t() @ U.t())

    pdb.set_trace()
    # Training kBit
    training_code = generate_code_new(train_data.cpu(), code_length, R, pca)
    k_bit_matrix_generator = KBitWeights(training_code, k, max_iter, logger, device)
    k_bit_matrix_generator.initialize_w()
    W = k_bit_matrix_generator.train()
        
    # Evaluate
    # Generate query code and retrieval code
    query_code = generate_code(query_data.cpu(), code_length, R, pca)
    retrieval_code = generate_code(retrieval_data.cpu(), code_length, R, pca)

    # Compute map
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
        topk,
    )

    # P-R curve
    P, Recall = pr_curve(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
    )

    # Save checkpoint
    checkpoint = {
        'qB': query_code,
        'rB': retrieval_code,
        'qL': query_targets,
        'rL': retrieval_targets,
        'pca': pca,
        'rotation_matrix': R,
        'P': P,
        'R': Recall,
        'map': mAP,
        'W': k_bit_matrix_generator.W,
    }

    return checkpoint


def generate_code(data, code_length, R, pca):
    """
    Generate hashing code.

    Args
        data(torch.Tensor): Data.
        code_length(int): Hashing code length.
        R(torch.Tensor): Rotration matrix.
        pca(callable): PCA function.

    Returns
        pca_data(torch.Tensor): PCA data.
    """
    return (torch.from_numpy(pca.transform(data.numpy())).to(R.device) @ R).sign()

def generate_code_new(data, code_length, R, pca):
    """
    Generate binary hashing code (0 or 1).

    Args
        data(torch.Tensor): Data.
        code_length(int): Hashing code length.
        R(torch.Tensor): Rotation matrix.
        pca(callable): PCA function.

    Returns
        binary_code(torch.Tensor): Binary code (0 or 1).
    """
    pca_data = torch.from_numpy(pca.transform(data.numpy())).to(R.device)
    continuous_code = pca_data @ R
    binary_code = (continuous_code >= 0).int()  # Convert to 0 or 1
    return binary_code
