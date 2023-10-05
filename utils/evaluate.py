import torch


def mean_average_precision_with_bit_similarity(query_code,
                                               retrieval_code,
                                               query_targets,
                                               retrieval_targets,
                                               device,
                                               bit_weights,
                                               topk=None):
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

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate bit similarity scores with bit weights influence
        bit_similarity_scores = query_code[i, :] * (bit_weights @ retrieval_code.t())

        # Arrange position according to bit similarity scores
        sorted_indices = torch.argsort(bit_similarity_scores, descending=True)
        retrieval = retrieval[sorted_indices][:topk]

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
    return mean_AP.item()


def mean_average_precision(query_code,
                           retrieval_code,
                           query_targets,
                           retrieval_targets,
                           device,
                           topk=None,
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
    return mean_AP.item()


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

    import matplotlib.pyplot as plt

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
