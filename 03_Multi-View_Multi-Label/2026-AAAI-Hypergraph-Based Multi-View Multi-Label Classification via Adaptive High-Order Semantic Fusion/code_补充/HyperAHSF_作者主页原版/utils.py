import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import defaultdict

def jaccard_unique_values(tensor):
    # Calculate intersection matrix (using matrix multiplication for acceleration)
    intersection = torch.mm(tensor, tensor.T)

    # Calculate the total number of labels for each sample
    row_sum = tensor.sum(dim=1)

    # Calculate union matrix (using broadcasting mechanism)
    union = row_sum.unsqueeze(1) + row_sum.unsqueeze(0) - intersection

    # Calculate Jaccard similarity, handling division by zero
    jaccard = torch.where(
        union != 0,
        intersection / union,
        torch.ones_like(intersection)  # Set to 1 for two empty sets
    )

    # Get all distinct Jaccard values
    unique_values = torch.unique(jaccard)
    print(unique_values)

def construct_hyperedge_index(tensors, thr, k):
    # Concatenate tensors from multiple views into matrix M
    M = torch.cat(tensors, dim=0)  # Shape [V*N, feature_dim]
    V = len(tensors)  # Number of views
    N = tensors[0].size(0)  # Number of samples per view
    device = M.device  # Ensure all tensors are on the same device

    # Build hyperedges based on prior knowledge
    prior_hyper_edges = []
    for i in range(N):
        nodes = [i + j * N for j in range(V)]
        prior_hyper_edges.append(nodes)

    # Initialize hyperedge_index, processing prior hyperedges
    hyperedge_list = []
    for e, nodes in enumerate(prior_hyper_edges):
        for n in nodes:
            hyperedge_list.append([n, e])
    current_e = len(prior_hyper_edges)  # Current hyperedge index, initial N

    # Calculate cosine similarity matrix
    M_normalized = torch.nn.functional.normalize(M, p=2, dim=1)
    sim_matrix = M_normalized @ M_normalized.T  # Shape [V*N, V*N]

    # Determine view for each node
    view_indices = torch.arange(V * N, device=device) // N  # Shape [V*N]

    # Build dynamic knn hyperedges
    for i in range(V * N):
        current_view = view_indices[i].item()
        # Exclude nodes from current view
        mask = view_indices != current_view
        candidates = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            continue  # Skip if no candidate nodes (may occur when V=1)

        # Get similarity for candidate nodes
        sim_i = sim_matrix[i]
        sim_candidates = sim_i[candidates]
        # Filter nodes with similarity >= threshold
        above_thr = sim_candidates >= thr
        selected = torch.nonzero(above_thr, as_tuple=False).squeeze(1)
        if selected.numel() == 0:
            continue

        # Sort by similarity in descending order
        sorted_scores, sorted_indices = torch.sort(sim_candidates[selected], descending=True)
        selected_candidates = candidates[selected[sorted_indices]]
        # Truncate to at most k nodes
        if selected_candidates.size(0) > k:
            selected_candidates = selected_candidates[:k]
        # Build hyperedge (including anchor i)
        hyperedge_nodes = torch.cat([torch.tensor([i], device=device), selected_candidates])
        # Add to hyperedge_list
        for n in hyperedge_nodes:
            hyperedge_list.append([n.item(), current_e])
        current_e += 1  # Increment hyperedge index

    # Convert to tensor and adjust shape
    if not hyperedge_list:
        hyperedge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        hyperedge_index = torch.tensor(hyperedge_list, dtype=torch.long, device=device).t().contiguous()

    return hyperedge_index

def construct_hyperedge(view_tensors, view_hyperedge_index):
    """
    view_tensors: List[torch.Tensor] V tensors of shape [N, feature_dim]
    view_hyperedge_index: List[torch.Tensor] Each element shape [2, hyperedge_connections]
    """
    V = len(view_tensors)
    N = view_tensors[0].size(0)
    device = view_tensors[0].device
    # Generate cross-view hyperedges
    views = torch.arange(V, device=device)  # (V,)
    samples = torch.arange(N, device=device)  # (N,)
    cross_nodes = (views[:, None] * N + samples[None, :]).T.reshape(-1)  # (V*N,)
    cross_edges = torch.repeat_interleave(torch.arange(N, device=device), V)
    cross_hyperedge = torch.stack([cross_nodes, cross_edges])  # (2, V*N)
    # Process intra-view hyperedges
    if V != len(view_hyperedge_index):
        raise ValueError("View count doesn't match view_hyperedge_index length")
    # Calculate hyperedge count per view
    e_counts = []
    for v in range(V):
        h_edge = view_hyperedge_index[v]
        if h_edge.numel() == 0:
            e_counts.append(0)
        else:
            max_e = h_edge[1].max()
            e_counts.append(max_e.item() + 1)
    # Calculate hyperedge start indices
    prefix_sums = torch.cumsum(torch.tensor([0] + e_counts, device=device), dim=0)
    start_indices = N + prefix_sums[:-1]  # (V,)
    # Convert hyperedges within each view
    view_hyperedges = []
    for v in range(V):
        h_edge = view_hyperedge_index[v]
        if h_edge.numel() == 0:
            continue
        # Convert node indices
        nodes = h_edge[0] + v * N
        # Convert hyperedge indices
        edges = h_edge[1] + start_indices[v]
        view_hyperedges.append(torch.stack([nodes, edges]))
    # Merge all hyperedges
    hyperedges = [cross_hyperedge]
    if view_hyperedges:
        hyperedges.append(torch.cat(view_hyperedges, dim=1))
    hyperedge_index = torch.cat(hyperedges, dim=1)

    return hyperedge_index

class PrecomputedContrastiveLoss(nn.Module):
    def __init__(self, Y, w_intra):
        """
        Initialize contrastive loss function
        Parameters:
            Y: Multi-label matrix of shape [num_samples, num_labels]
        """
        super().__init__()
        self.valid_group_indices = []  # Store valid group indices
        self.group_sizes = []  # Store sample count per group
        self.w_intra = w_intra

        # 1. Preprocess label data, create group indices
        label_dict = defaultdict(list)
        for i, label_vec in enumerate(Y):
            # Convert label vector to hashable tuple
            key = tuple(label_vec.tolist())
            label_dict[key].append(i)

        # 2. Filter valid groups (k>=2) and store indices
        for indices in label_dict.values():
            k = len(indices)
            if k >= 2:
                self.valid_group_indices.append(indices)
                self.group_sizes.append(k)

        # 3. Convert index list to tensor for GPU acceleration
        self.group_tensors = []
        for indices in self.valid_group_indices:
            # Convert to LongTensor and register as buffer
            idx_tensor = torch.LongTensor(indices)
            self.register_buffer(f'group_{id(idx_tensor)}', idx_tensor)
            self.group_tensors.append(idx_tensor)

        self.num_groups = len(self.valid_group_indices)

        # 4. Print warning if no valid groups
        if self.num_groups == 0:
            print("Warning: No valid groups found (all groups have size < 2).")

    def forward(self, X):
        """
        Calculate contrastive loss with negative samples
        Parameters:
            X: Feature tensor of shape [num_samples, feature_dim]
        Returns:
            Scalar loss value
        """
        # Return 0 loss if no valid groups
        if self.num_groups == 0:
            return torch.tensor(0.0, device=X.device)

        # Normalize entire feature matrix (improves efficiency)
        X_normalized = F.normalize(X, p=2, dim=1)  # [num_samples, feature_dim]
        total_samples = X.size(0)
        total_loss = torch.tensor(0.0, device=X.device)

        # 5. Iterate through all valid groups
        for i, indices in enumerate(self.group_tensors):
            # Ensure indices on correct device
            indices = indices.to(X.device)
            k = self.group_sizes[i]

            # 6. Extract current group features
            group_features = X_normalized[indices]  # [k, feature_dim]

            # 7. Calculate class prototype (mean vector) and normalize
            prototype = group_features.mean(dim=0, keepdim=True)  # [1, feature_dim]
            prototype = F.normalize(prototype, p=2, dim=1)  # Normalize prototype

            # 8. Calculate cosine similarity between all samples and current prototype
            all_similarities = torch.mm(X_normalized, prototype.t()).squeeze(1)  # [num_samples]

            # 9. Positive term: Intra-group similarity (negative)
            pos_similarities = all_similarities[indices]  # [k]
            pos_term = -pos_similarities.mean()  # Maximize intra-group similarity

            # 10. Negative term: Average similarity of non-group samples
            num_neg = total_samples - k
            if num_neg > 0:
                # Create mask to select non-group samples
                mask = torch.ones(total_samples, dtype=torch.bool, device=X.device)
                mask[indices] = False
                neg_similarities = all_similarities[mask]  # [num_neg]
                neg_term = neg_similarities.mean()  # Minimize inter-group similarity
            else:
                neg_term = torch.tensor(0.0, device=X.device)

            # 11. Combine loss terms: Positive + Negative
            group_loss = pos_term + neg_term
            total_loss += group_loss

        # 12. Calculate average loss
        return self.w_intra * total_loss / self.num_groups