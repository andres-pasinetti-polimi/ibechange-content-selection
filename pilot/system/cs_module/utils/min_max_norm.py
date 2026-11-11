def min_max_norm(x, x_min, x_max):
    norm = (x - x_min) / (x_max - x_min)
    return norm
