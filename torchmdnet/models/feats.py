# Based on the code from: https://github.com/klicperajo/dimenet,
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet_utils.py

import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import torch
from math import pi as PI
from torch_sparse import SparseTensor
from torch_scatter import scatter
import sympy as sym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Jn(r, n):
    """
    Compute the spherical Bessel function of the first kind.

    Parameters:
    r (float or array-like): The value(s) at which to evaluate the Bessel function.
    n (int): The order of the Bessel function.

    Returns:
    float or ndarray: The value(s) of the spherical Bessel function of the first kind at r.
    """
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical Bessel function of the first kind for orders 0 to n-1.

    Parameters:
    n (int): The number of orders of the spherical Bessel function.
    k (int): The number of zeros to compute for each order.

    Returns:
    numpy.ndarray: A 2D array of shape (n, k) containing the first k zeros for each order from 0 to n-1.
    """
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i, ))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Generate the first n spherical Bessel functions of the first kind using symbolic differentiation.

    Parameters:
    n (int): The number of spherical Bessel functions to generate.

    Returns:
    list: A list of n symbolic expressions representing the spherical Bessel functions of the first kind.
    """
    x = sym.symbols('x')

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x)**i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """
    Generate a basis of normalized spherical Bessel functions.

    Parameters:
    n (int): The number of orders of the spherical Bessel functions.
    k (int): The number of zeros to use for normalization of each order.

    Returns:
    list: A list of lists containing symbolic expressions of the normalized spherical Bessel functions for each order and zero.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(k, m):
    """
    Compute the prefactor for spherical harmonics.

    Parameters:
    k (int): The degree of the spherical harmonic.
    m (int): The order of the spherical harmonic.

    Returns:
    float: The prefactor for the spherical harmonic of degree k and order m.
    """
    return ((2 * k + 1) * np.math.factorial(k - abs(m)) /
            (4 * np.pi * np.math.factorial(k + abs(m))))**0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    """
    Compute associated Legendre polynomials.

    Parameters:
    k (int): The maximum degree of the associated Legendre polynomials.
    zero_m_only (bool, optional): If True, only compute the polynomials for m=0. Defaults to True.

    Returns:
    list: A nested list containing the associated Legendre polynomials for each degree and order up to k.
    """
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] -
                                        (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))

    return P_l_m


def real_sph_harm(l, zero_m_only=False, spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        x = sym.symbols('x')
        y = sym.symbols('y')
        S_m = [x*0]
        C_m = [1+0*x]
        # S_m = [0]
        # C_m = [1]
        for i in range(1, l):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x*S_m[i-1] + y*C_m[i-1]]
            C_m += [x*C_m[i-1] - y*S_m[i-1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))

    Y_func_l_m = [['0']*(2*j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        """
        Envelope function to scale inputs using a polynomial of a specified exponent.

        Parameters:
        exponent (int): The exponent for the polynomial.
        """
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        """
        Apply the envelope function to the input tensor.

        Parameters:
        x (torch.Tensor): Input tensor to be scaled.

        Returns:
        torch.Tensor: Scaled input tensor using the envelope function.
        """
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class dist_emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        """
        Distance embedding module with a cutoff and envelope function.

        Parameters:
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for the embedding. Defaults to 5.0.
        envelope_exponent (int, optional): Exponent for the envelope function. Defaults to 5.
        """
        super(dist_emb, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        self.freq.data = torch.arange(1, self.freq.numel() + 1).float().mul_(PI)

    def forward(self, dist):
        """
        Apply the distance embedding to the input distances.

        Parameters:
        dist (torch.Tensor): Input distance tensor.

        Returns:
        torch.Tensor: Transformed distance tensor after applying the envelope function and sinusoidal radial basis functions.
        """
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class angle_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        
        """
        A module for computing angular embeddings using spherical harmonics and Bessel functions.

        Args:
            num_spherical (int): Number of spherical harmonics to consider.
            num_radial (int): Number of radial basis functions to consider, should be <= 64.
            cutoff (float, optional): Cutoff radius for the embeddings. Default is 5.0.
            envelope_exponent (int, optional): Exponent for envelope function (not currently used).

        Attributes:
            num_spherical (int): Number of spherical harmonics.
            num_radial (int): Number of radial basis functions.
            cutoff (float): Cutoff radius for the embeddings.
            sph_funcs (list): List of functions for spherical harmonics.
            bessel_funcs (list): List of functions for Bessel basis functions.

        Raises:
            AssertionError: If num_radial exceeds 64.

        Notes:
            This module uses symbolic computation to generate basis functions for spherical harmonics
            and Bessel functions and converts them into torch-compatible functions.

        """
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        """
        Forward pass method for computing angular embeddings.

        Args:
            dist (torch.Tensor): Distance tensor normalized by the cutoff radius.
            angle (torch.Tensor): Angle tensor.
            idx_kj (torch.Tensor): Indices for selecting specific basis functions.

        Returns:
            torch.Tensor: Tensor containing computed angular embeddings.

        """
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class torsion_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        """
        A module for computing torsional embeddings using spherical harmonics and Bessel functions.

        Args:
            num_spherical (int): Number of spherical harmonics to consider.
            num_radial (int): Number of radial basis functions to consider, should be <= 64.
            cutoff (float, optional): Cutoff radius for the embeddings. Default is 5.0.
            envelope_exponent (int, optional): Exponent for envelope function (not currently used).

        Attributes:
            num_spherical (int): Number of spherical harmonics.
            num_radial (int): Number of radial basis functions.
            cutoff (float): Cutoff radius for the embeddings.
            sph_funcs (list): List of functions for spherical harmonics.
            bessel_funcs (list): List of functions for Bessel basis functions.

        Raises:
            AssertionError: If num_radial exceeds 64.

        Notes:
            This module uses symbolic computation to generate basis functions for spherical harmonics
            and Bessel functions and converts them into torch-compatible functions. It supports
            spherical harmonics with non-zero m values.

        """
        super(torsion_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: torch.zeros_like(x) + torch.zeros_like(y) + sph1(0,0)) #torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k+i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        """
        Forward pass method for computing torsional embeddings.

        Args:
            dist (torch.Tensor): Distance tensor normalized by the cutoff radius.
            angle (torch.Tensor): Angle tensor (theta).
            phi (torch.Tensor): Angle tensor (phi).
            idx_kj (torch.Tensor): Indices for selecting specific basis functions.

        Returns:
            torch.Tensor: Tensor containing computed torsional embeddings.

        """
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        return out

def xyz_to_dat(pos, edge_index, num_nodes, use_torsion = False):
    """
    Compute the diatance, angle, and torsion from geometric information.

    Args:
        pos: Geometric information for every node in the graph.
        edge_index: Edge index of the graph.
        number_nodes: Number of nodes in the graph.
        use_torsion: If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)
    """
    j, i = edge_index  # j->i

    # Calculate distances. # number of edges
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Calculate angles. 0 to pi
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)


    if use_torsion:
        # Prepare torsion idxes.
        idx_batch = torch.arange(len(idx_i),device=device)
        idx_k_n = adj_t[idx_j].storage.col()
        repeat = num_triplets
        num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
        idx_i_t = idx_i.repeat_interleave(num_triplets_t)
        idx_j_t = idx_j.repeat_interleave(num_triplets_t)
        idx_k_t = idx_k.repeat_interleave(num_triplets_t)
        idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
        mask = idx_i_t != idx_k_n   
        idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

        # Calculate torsions.
        pos_j0 = pos[idx_k_t] - pos[idx_j_t]
        pos_ji = pos[idx_i_t] - pos[idx_j_t]
        pos_jk = pos[idx_k_n] - pos[idx_j_t]
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(pos_ji, pos_j0)
        plane2 = torch.cross(pos_ji, pos_jk)
        a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji 
        torsion1 = torch.atan2(b, a) # -pi to pi
        torsion1[torsion1<=0]+=2*PI # 0 to 2pi
        torsion = scatter(torsion1,idx_batch_t,reduce='min')

        return dist, angle, torsion, i, j, idx_kj, idx_ji
    
    else:
        return dist, angle, i, j, idx_kj, idx_ji

