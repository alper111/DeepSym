import torch


def kmeans(x, k, centroids=None, max_iter=None, epsilon=0.01):
    '''
    x: data set of size (n, d) where n is the sample size.
    k: number of clusters
    centroids (optional): initial centroids
    max_iter (optional): maximum number of iterations
    epsilon (optional): error tolerance
    returns
    centroids: centroids found by k-means algorithm
    next_assigns: assignment vector
    mse: mean squared error
    '''
    if centroids is None:
        centroids = torch.zeros(k, x.shape[1], device=x.device)
        prev_assigns = torch.randint(0, k, (x.shape[0],), device=x.device)
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)

    distances = torch.cdist(centroids, x) ** 2
    prev_assigns = torch.argmin(distances, dim=0)

    it = 0
    prev_mse = distances[prev_assigns, torch.arange(x.shape[0])].mean()
    while True:
        for i in range(k):
            if (prev_assigns == i).sum() > 0:
                centroids[i] = x[prev_assigns == i].mean(dim=0)
        distances = torch.cdist(centroids, x) ** 2
        next_assigns = torch.argmin(distances, dim=0)
        if (next_assigns == prev_assigns).all():
            break
        else:
            prev_assigns = next_assigns
        it += 1
        mse = distances[next_assigns, torch.arange(x.shape[0])].mean()
        error = abs(prev_mse-mse)/prev_mse
        prev_mse = mse
        print("iteration: %d, mse: %.3f" % (it, prev_mse.item()))

        if it == max_iter:
            break
        if error < epsilon:
            break

    return centroids, next_assigns, prev_mse, it
