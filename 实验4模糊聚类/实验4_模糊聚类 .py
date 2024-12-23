import numpy as np

np.set_printoptions(precision=2)  # 设置矩阵输出精度，保留两位小数


def normalize_by_max(matrix):
    """
    使用最大值归一化方法，将矩阵每列归一化。
    """
    return matrix / matrix.max(axis=0, keepdims=True)


def construct_fuzzy_similarity_matrix(matrix):
    """
    使用最大最小法构造模糊相似矩阵。
    """
    normalized = normalize_by_max(matrix)
    size = normalized.shape[0]
    similarity_matrix = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            row_union = np.maximum(normalized[i], normalized[j])
            row_intersection = np.minimum(normalized[i], normalized[j])
            similarity_matrix[i, j] = row_intersection.sum() / row_union.sum()

    return similarity_matrix


def fuzzy_matrix_composition(matrix_a, matrix_b):
    """
    进行模糊矩阵的合成。
    """
    rows, cols = matrix_a.shape[0], matrix_b.shape[1]
    composed_matrix = np.zeros((rows, cols), dtype=float)

    for i in range(rows):
        for j in range(cols):
            composed_matrix[i, j] = max(min(matrix_a[i, k], matrix_b[k, j]) for k in range(matrix_a.shape[1]))

    return composed_matrix


def compute_transitive_closure(matrix):
    """
    使用平方法计算传递闭包。
    """
    closure = construct_fuzzy_similarity_matrix(matrix)
    while True:
        new_closure = fuzzy_matrix_composition(closure, closure)
        if np.allclose(new_closure, closure):
            return np.round(new_closure, 2)
        closure = new_closure


def compute_cut_set(matrix):
    """
    计算水平截集。
    """
    closure = compute_transitive_closure(matrix)
    unique_values = np.unique(closure)
    return np.sort(unique_values)[::-1]


def extract_classes(pairs, size):
    """
    根据给定的元素对提取聚类类集。
    """
    clusters = [set() for _ in range(size)]
    for x, y in pairs:
        clusters[x].add(y)
        clusters[y].add(x)

    return [list(cluster) for cluster in clusters if cluster]


def perform_fuzzy_clustering(matrix):
    """
    执行模糊聚类并返回聚类结果。
    """
    cut_levels = compute_cut_set(matrix)
    closure = compute_transitive_closure(matrix)
    size = matrix.shape[0]

    clustering_results = []
    for level in cut_levels:
        if level == cut_levels[0]:
            clustering_results.append([[i] for i in range(size)])
        else:
            element_pairs = np.argwhere(closure >= level)
            clustering_results.append(extract_classes(element_pairs, size))

    return clustering_results


def main():
    """
    主函数，演示模糊聚类过程。
    """
    data_matrix = np.array([[17, 15, 14, 15, 16],
                            [18, 16, 13, 14, 12],
                            [18, 18, 19, 17, 18],
                            [16, 18, 16, 15, 18]])

    print("特性指标矩阵:\n", data_matrix)
    print("\n归一化矩阵:\n", normalize_by_max(data_matrix))
    print("\n模糊相似矩阵:\n", construct_fuzzy_similarity_matrix(data_matrix))
    print("\n传递闭包:\n", compute_transitive_closure(data_matrix))
    print("\n水平截集:\n", compute_cut_set(data_matrix))
    print("\n模糊聚类结果:\n", perform_fuzzy_clustering(data_matrix))


if __name__ == "__main__":
    main()
