def cost(q, p, l, v):
    w = q * l + p
    z = (w - v) ** 2
    return z


def fun(x, a, b):
    y = a * x + b
    return y


def slope(a, b, c, d):
    y = (d - c)
    z = (b - a)
    m = y / z
    return m

