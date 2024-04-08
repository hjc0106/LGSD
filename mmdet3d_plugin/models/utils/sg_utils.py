import torch

def knn_query(k, xyz, query_xyz):
    """
    Input:
        ngroup: max group number in local region
        xyz: all points/voxels features, [B, N, C]
        query_xyz: query points/voxels features, [B, S, C]
    Return:
        group_idx: grouped points/voxels index, [B, S, ngroup]
    """
    B, N, _ = query_xyz.shape
    _, M, _ = xyz.shape
    dist = -2 * torch.matmul(query_xyz, xyz.permute(0, 2, 1))
    dist += torch.sum(query_xyz ** 2, -1).view(B, N, 1)
    dist += torch.sum(xyz ** 2, -1).view(B, 1, M)
    _, group_idx = torch.topk(dist, k, dim=-1, largest=False, sorted=False)
    
    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Returns:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    
    return new_points
