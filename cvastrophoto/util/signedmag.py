
def min_directed(dir_val, clamp_val):
    if dir_val > 0:
        return min(dir_val, max(clamp_val, 0))
    elif dir_val < 0:
        return max(dir_val, min(clamp_val, 0))
    else:
        return 0
