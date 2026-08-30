for i in range(1, 100):
    for j in range(1, 100):
        for k in range(1, 100):
            if (1 / i + 1 / j + 1 / k > 0.999) and (1 / i + 1 / j + 1 / k < 1.001):
                print(i, j, k, (1/i + 1/j + 1/k))