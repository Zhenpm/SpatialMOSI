import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import faiss

def coor_transform(coor, M):
    return np.dot(M, np.hstack((coor, np.array([1] * coor.shape[0]).reshape(-1, 1))).T)

def find_similar_index(source: np.ndarray, target: np.ndarray, top_k: int=1):
    index = faiss.IndexFlatL2(source.shape[1])

    faiss.normalize_L2(source)
    faiss.normalize_L2(target) 
    index.add(target)

    return index.search(source, top_k)

def get_transform(adata_list, dst_id, src_id_list, target_label_id, threshold=50, max_iterations=5000, tolerance=1e-3, show_plot=True, seed=0):
    
    transform_matrix = {}

    for src_id in src_id_list:
        if (src_id == dst_id):
            transform_matrix[src_id] = np.eye(3)
            continue

        dst_coor = adata_list[dst_id][target_label_id == adata_list[dst_id].obs['mclust']].obsm['spatial']
        src_coor = adata_list[src_id][target_label_id == adata_list[src_id].obs['mclust']].obsm['spatial']

        print(f'>>> INFO: dst slice id: {dst_id}, src slice id: {src_id}')
        print('>>> INFO: dst coordination shape:', dst_coor.shape)
        print('>>> INFO: src coordination shape:', src_coor.shape)

        T = icp(src_coor, dst_coor, threshold, max_iterations, tolerance, seed)
        src_coor = coor_transform(src_coor, T).T
        transform_matrix[src_id] = T
        if show_plot:
            plt.scatter(x=dst_coor[:, 0], y=dst_coor[:, 1], label='dst point cloud')
            plt.scatter(x=src_coor[:, 0], y=src_coor[:, 1], label='src point cloud')
            plt.show()

    return transform_matrix

def icp(src, dst, threshold=50, max_iterations=5000, tolerance=1e-3, seed=0):

    # sample spots from two sets to the same number
    np.random.seed(seed)
    if (dst.shape[0] > src.shape[0]):
        src = src.astype(np.float32)
        dst = dst[np.random.choice(dst.shape[0], src.shape[0], replace=False)].astype(np.float32)
    elif (dst.shape[0] < src.shape[0]):
        src = src[np.random.choice(src.shape[0], dst.shape[0], replace=False)].astype(np.float32)
        dst = dst.astype(np.float32)

    # init
    cur_src = np.hstack((src, np.array([1] * src.shape[0]).reshape(-1, 1))).T
    prev_error = 0

    # train ICP
    for _ in range(max_iterations):
        distances, indices = nearest_neighbor(cur_src[:2, :].T, dst)
        M, _, _ = best_fit_transform(cur_src[:2, :].T, dst[indices])
        cur_src = coor_transform(cur_src[:2, :].T, M)

        mean_error = np.mean(distances)
        if (np.abs(prev_error - mean_error) < tolerance):
            # stuck in local optimum -> rotate src
            if (threshold < mean_error):
                rotate_deg = np.pi * 2 / 3
                cur_src = coor_transform(cur_src[:2, :].T, np.array([
                    [np.cos(rotate_deg), np.sin(rotate_deg), 0.5], 
                    [-np.sin(rotate_deg), np.cos(rotate_deg), 0.5], 
                    [0, 0, 1]
                ]))
            else:
                break
        prev_error = mean_error

    print(f'>>> INFO: current distance: {mean_error}')
    M, _, _ = best_fit_transform(src, cur_src[:2,:].T)
    return M


# ICP algorithm from https://github.com/ClayFlannigan/icp/blob/master/icp.py
def nearest_neighbor(src, dst):

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t