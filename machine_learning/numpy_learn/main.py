import numpy as np

if __name__ == '__main__':
    # 2x3 matrix
    mat = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    # Num of rows and cols. Dimensions of matrix
    print(f"Dimensions: {mat.shape}")

    print(f"Size: {mat.size}")
    print(f"Data type: {mat.dtype}")

    print(f"Transpose of the matrix: {mat.T}")
    print(f"Mean (avg): {mat.mean()}")
    print(f"Min: {mat.min()}, Max: {mat.max()}")
    print(f"Index of min value: {mat.argmin()}")
    print(f"Random 3x3 matrix: {np.random.rand(3, 3)}")
    print(f"Ones matrix: {np.ones((3, 3), int)}")
