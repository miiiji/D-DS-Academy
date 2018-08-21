import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import elice_utils

def main():
    plt.figure(figsize=(5,5)) #5인치*5인치의 넓이의 정사각형
    
    X = []
    Y = []
    # N을 10배씩 증가할 때 파이 값이 어떻게 변경되는지 확인해보세요.
    N = 100000 #점점 파이값이 3.14에 가까워진다
    for i in range(N):
        X.append(np.random.random())
        Y.append(np.random.random())
    X = np.array(X)
    Y = np.array(Y)
    X = X * 2 - 1 # [0,1] --> [0,2] --> [-1,1]
    Y = Y * 2 - 1
    dist = np.sqrt(X ** 2 + Y ** 2)
    is_inside_circle= dist <= 1
    
    #     for i in range(N):
    #         X.append(np.random.rand() * 2 - 1)
    #         Y.append(np.random.rand() * 2 - 1)
    #     X = np.array(X)
    #     Y = np.array(Y)
    #     distance_from_zero = np.sqrt(X * X + Y * Y)
    #     is_inside_circle = distance_from_zero <= 1
    
    print("Estimated pi = %f" % (np.average(is_inside_circle) * 4))
    
    plt.scatter(X, Y, c=is_inside_circle)
    plt.savefig('circle.png')
    elice_utils.send_image('circle.png')

if __name__ == "__main__":
    main()
