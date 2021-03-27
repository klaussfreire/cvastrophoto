from . import gaussian


def gaussian_decompose(img, nlevels):
    if nlevels == 1:
        return [img]

    levels = [img]
    for i in range(nlevels - 1):
        nextlevel = gaussian.fast_gaussian(levels[-1], 2 ** (i + 1))
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