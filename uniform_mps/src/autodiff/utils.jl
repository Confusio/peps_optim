"""Autodiff helper utilities for UniformMPS."""

function wirtinger(f, z0)
    y, back = Zygote.pullback(f, z0)
    du, dv = back(1)[1], back(im)[1]
    dfdz = (du' + im * dv') / 2
    dfdzbar = (du + im * dv) / 2
    return dfdz, dfdzbar
end
