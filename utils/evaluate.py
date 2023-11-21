import torch
import matplotlib.pyplot as plt
import pdb
import numpy as np


def compute_hamming_distance(query_code, retrieval_tensor):
    # Calculate Hamming distance
    hamming_distances = (query_code.unsqueeze(0) != retrieval_tensor).sum(dim=1)

    return hamming_distances

def filter_retrieval_tensor(query_code, retrieval_tensor, max_hamming_distance=1):
    # Compute Hamming distances
    hamming_distances = compute_hamming_distance(query_code, retrieval_tensor)

    # Find indices where Hamming distance is less than or equal to max_hamming_distance
    valid_indices = (hamming_distances <= max_hamming_distance).nonzero().squeeze()

    # Filter retrieval tensor and record original indices
    filtered_retrieval_tensor = retrieval_tensor[valid_indices]

    return filtered_retrieval_tensor, valid_indices


def mean_average_precision_with_bit_similarity(query_code,
                                               retrieval_code,
                                               query_targets,
                                               retrieval_targets,
                                               device,
                                               bit_weights,
                                               topk=None,
                                               k=1):
    """
    Calculate mean average precision (MAP) with bit similarity scores and influence of bit weights.

    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Retrieval data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot.
        retrieval_targets (torch.Tensor): Retrieval data targets, one-hot.
        device (torch.device): Using CPU or GPU.
        bit_weights (torch.Tensor): Bit weight matrix of size (code_length x code_length).
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_targets.shape[0]
    mean_AP = 0.0
    pdb.set_trace()
    hamm = 0.0
    # bit_weights = torch.diag(bit_weights)
    for i in range(num_query):
        # Retrieve images from database
        
        query_code = query_code.double()
        retrieval_code = retrieval_code.double()

        retrieval_code, valid_ret_indices = filter_retrieval_tensor(query_code[i, :], retrieval_code, k)

        retrieval_targets = retrieval_targets[valid_ret_indices]

        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # weighted_hamming_distances = torch.sum(bit_weights * (query_code[i, :] != retrieval_code), dim=1)

        # Generate similarity scores based on the distances (lower distance => higher score)
        # scores = 1 / (1 + weighted_hamming_distances)

        # sorted_indices = torch.argsort(scores, descending=True)
        
        # # bit_weights = torch.from_numpy(bit_weights).double().to(query_code.device)

        # Calculate bit similarity scores with bit weights influence
        # bit_similarity_scores = (query_code[i, :] + retrieval_code) * bit_weights
        # bit_similarity_scores = torch.abs(bit_similarity_scores)

        # bit_similarity_scores = (torch.diag(bit_weights) * query_code[i, :]).unsqueeze(dim=0) @ retrieval_code.T
        # bit_similarity_scores = bit_similarity_scores.squeeze(dim=0)

        # bit_similarity_scores = torch.sum(bit_weights * (query_code[i, :] != retrieval_code), dim=1)
        # scores = 1 / (1 + bit_similarity_scores)

        # hamming_dist = 0.5 * (retrieval_code.shape[1] - (torch.sum(bit_weights, dim=1) * query_code[i, :]).unsqueeze(dim=0) @ retrieval_code.T)
        xor_q_ret = torch.bitwise_xor(query_code[i, :].to(torch.long), retrieval_code.to(torch.long)).to(torch.double)
        # hamming_dist = 0.5 * (torch.sum(bit_weights, dim=1) @ xor_q_ret.T)
        hamming_dist = 0.5 * (bit_weights @ xor_q_ret.T)
        # hamm += hamming_dist.mean().item()
        # bit_similarity_scores = bit_similarity_scores.sum(dim=1)

        # sorted_scores_row_wise, _ = torch.sort(bit_similarity_scores, dim=1, descending=True)
        # sorted_scores_row_wise = sorted_scores_row_wise[:, 0]

        # Arrange position according to bit similarity scores
        # sorted_indices = torch.argsort(sorted_scores_row_wise, descending=True)
        # sorted_indices = torch.argsort(bit_similarity_scores, descending=True)
        retrieval = retrieval[torch.argsort(hamming_dist.squeeze())][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float().reshape(-1)

        # index = index.topk(score.shape[0]).indices

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    # hamm = hamm / num_query
    # print(hamm)
    return (mean_AP.item() if (type(mean_AP) is torch.Tensor) else mean_AP)


def mean_average_precision(query_code,
                           retrieval_code,
                           query_targets,
                           retrieval_targets,
                           device,
                           topk=None,
                           k=1
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Retrieval data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        retrieval_targets (torch.Tensor): retrieval data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval_code, valid_ret_indices = filter_retrieval_tensor(query_code[i, :], retrieval_code, k)

        retrieval_targets = retrieval_targets[valid_ret_indices]

        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return (mean_AP.item() if (type(mean_AP) is torch.Tensor) else mean_AP)


def pr_curve(query_code, retrieval_code, query_targets, retrieval_targets, device):
    """
    P-R curve.

    Args
        query_code(torch.Tensor): Query hash code.
        retrieval_code(torch.Tensor): Retrieval hash code.
        query_targets(torch.Tensor): Query targets.
        retrieval_targets(torch.Tensor): Retrieval targets.
        device (torch.device): Using CPU or GPU.

    Returns
        P(torch.Tensor): Precision.
        R(torch.Tensor): Recall.
    """
    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = torch.zeros(num_query, num_bit + 1).to(device)
    R = torch.zeros(num_query, num_bit + 1).to(device)
    for i in range(num_query):
        gnd = (query_targets[i].unsqueeze(0).mm(retrieval_targets.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask

    P = P.cpu().numpy()
    R = R.cpu().numpy()

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(R, P, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (128 bit)')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'PR-curve-{num_bit}.jpg')

    return P, R


def pr_curve_bit_similarity(query_code, retrieval_code, query_targets, retrieval_targets, device, bit_weights):
    """
    P-R curve based on bit similarity scores with bit weights influence.

    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Retrieval data hash code.
        query_targets (torch.Tensor): Query data targets.
        retrieval_targets (torch.Tensor): Retrieval data targets.
        device (torch.device): Using CPU or GPU.
        bit_weights (torch.Tensor): Bit weight matrix of size (code_length x code_length).

    Returns:
        P (torch.Tensor): Precision.
        R (torch.Tensor): Recall.
    """
    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = torch.zeros(num_query).to(device)
    R = torch.zeros(num_query).to(device)
    query_code = query_code.double()
    retrieval_code = retrieval_code.double()

    for i in range(num_query):
        # pdb.set_trace()
        # Calculate bit similarity scores with bit weights influence
        bit_similarity_scores = query_code[i, :] * (bit_weights @ retrieval_code.t()).t()

        # Initialize retrieval tensor with zeros
        retrieval = torch.zeros_like(retrieval_targets, dtype=torch.float)

        # Update retrieval based on the bit_similarity_scores
        for bit_index in range(min(num_bit, retrieval.shape[1])):
            retrieval[:, bit_index] = (bit_similarity_scores[:, bit_index] >= bit_similarity_scores[:, num_bit - bit_index - 1]).float()

        #  Calculate True Positives and False Positives
        true_positives = (retrieval * retrieval_targets).sum(dim=1)
        false_positives = (retrieval * (1 - retrieval_targets)).sum(dim=1)

        # Avoid dividing by zero
        denominator = true_positives + false_positives
        valid_indices = denominator != 0

        # Calculate Precision and Recall
        precision = torch.zeros_like(denominator, dtype=torch.float)
        precision[valid_indices] = true_positives[valid_indices] / denominator[valid_indices]
        recall = true_positives / retrieval_targets.sum(dim=1)

        # pdb.set_trace()

        # Average precision and recall for all queries
        P[i] = precision.mean()
        R[i] = recall.mean()

    P = P.cpu().numpy()
    R = R.cpu().numpy()

    # P = min_max_scaling(P)
    # R = min_max_scaling(R)

    # pdb.set_trace()

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(R, P, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Bit Similarity with Bit Weights)')
    plt.grid(True)
    plt.xlim(0.4, 0.6)
    plt.ylim(0, 0.4)
    plt.savefig(f'PR-curve-bit-similarity-{num_bit}.jpg')

    return P, R

def min_max_scaling(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (array - min_val) / (max_val - min_val)
    return scaled_array
