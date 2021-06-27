from . import gaussian
import scipy.ndimage


def gaussian_decompose(img, nlevels, scale=1, tophat=True):
    if nlevels == 1:
        return [img]

    levels = [img.copy()]
    for i in range(nlevels - 1):
        size = scale * (2 ** (i + 1))
        level_data = levels[-1]
        if tophat:
            level_data = scipy.ndimage.minimum_filter(level_data, size)
        nextlevel = gaussian.fast_gaussian(level_data, size)
        levels[-1] -= nextlevel
        levels.append(nextlevel)

    return levels


def gaussian_recompose(levels, copy=False):
    img = levels[0]
    if copy:
        img = img.copy()
    for level in levels[1:]:
        img += level
    return img