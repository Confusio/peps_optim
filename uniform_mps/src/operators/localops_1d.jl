"""Spin-1 自旋算符集合 (Sx, Sy, Sz)."""
function spin1_ops()
    s = 1 / sqrt(2)
    Sx = s * [0 1 0; 1 0 1; 0 1 0]
    Sy = s * [0 -im 0; im 0 -im; 0 im 0]
    Sz = Diagonal([1, 0, -1])
    return Sx, Sy, Sz
end

"""最近邻双站点哈密顿量（默认为各向同性 Heisenberg）。"""
function two_site_h(; J::Real=1.0)
    Sx, Sy, Sz = spin1_ops()
    h_mat = J * (kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz))
    TensorMap(h_mat, P ⊗ P, P ⊗ P)
end

