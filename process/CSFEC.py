from numpy import pi, absolute, sum, exp, arange, zeros, array, delete, where, finfo, float32, argmax, cos, sin,log
from numpy.fft import fft
eps = finfo(float32).eps


class const:
    def __init__(self, K, N, M, NCSFE=5, Nya=0, minmaxidx=None):
        """

        :param K:
        :param N:
        :param M:
        :param NCSFE:
        :param Nya:
        :param minmaxidx:
        """
        if minmaxidx is not None:
            self.min_idx = minmaxidx[0]
            self.max_idx = minmaxidx[1]
        else:
            self.min_idx = 0
            self.max_idx = -1
        self.K = K
        self.N = N
        self.N0 = self.N * M
        self.V0 = arange(self.N0) / self.N0
        self.NCSFE = max(1, NCSFE)
        self.n = arange(self.N)
        self.seq = arange(self.N0)
        self.Nya = Nya


def residualDFT_new(q, const):
    N = const.N
    N2 = N ** 2

    qb = q ** N
    qc = q ** (N + 1)
    qd = q ** (N + 2)
    q_1 = q - 1
    den0 = q_1
    den1 = q_1 ** 2
    den2 = q_1 ** 3

    problems = where(absolute(q-1) < eps)
    den0[problems] = 1
    den1[problems] = 1
    den2[problems] = 1

    qb[problems] = N + 1
    qc[problems] = (3*N2+N-2)/(N-1)
    qd[problems] = (((2*N-1)*(N-1)*N)/6 -
                    (-2 * N2 + 2 * N + 1) * qc[problems]
                    + N2 * qb[problems] - 2)/((N - 1) ** 2)

    F0 = (qb - 1) / (den0*N)
    F1 = ((N - 1) * qc - N * qb + q) / (den1*N)
    F2 = (((N - 1) ** 2) * qd + (-2 * N2 + 2 * N + 1) * qc + N2 * qb - q ** 2 - q) / (den2*N)

    return [F0, F1, F2]


def residualDFT0(q, const):
    N = const.N
    qb = q ** N
    q_1 = q - 1
    den0 = q_1
    problems = where(absolute(q-1) < eps)
    den0[problems] = 1
    qb[problems] = N + 1
    F0 = (qb - 1) / (den0*N)
    return F0


def CSFEC(x0, cnst: const):
    x1 = x0 * arange(cnst.N)
    x2 = x1 * arange(cnst.N)
    x = array([x0, x1, x2])

    X = fft(x0, n=cnst.N0, axis=0) / cnst.N

    A = zeros((cnst.K, 1), dtype=complex)
    f = zeros((cnst.K, 1))
    X0 = X
    for k in range(cnst.K):
        alph = argmax(absolute(X0[cnst.min_idx:cnst.max_idx])) + cnst.min_idx
        A[k] = X0[alph]
        f[k] = cnst.V0[alph]

        ord = arange(k)
        for jj in ord:
            F = f[jj]
            idx = delete(ord, jj)

            for it in range(cnst.NCSFE):
                fdiff = f[idx] - F
                q = exp(1j * 2 * pi * fdiff)
                F0, F1, F2 = residualDFT_new(q, cnst)

                AF0_local = sum(A[idx] * F0, axis=0)
                AF1_local = sum(A[idx] * F1, axis=0)
                AF2_local = sum(A[idx] * F2, axis=0)

                C = 1 / cnst.N * (sum(x[0, :] * exp(-1j * 2 * pi * cnst.n * F))) - AF0_local
                X1 = 1 / cnst.N * (sum(x[1, :] * exp(-1j * 2 * pi * cnst.n * F))) - AF1_local
                X2 = 1 / cnst.N * (sum(x[2, :] * exp(-1j * 2 * pi * cnst.n * F))) - AF2_local

                b = (C.conjugate() * X2).real
                c = -(C.conjugate() * X1).imag

                Delta = -c / b
                F = F + Delta / (2 * pi)

            f[jj] = F
            A[jj] = C

        idx2 = arange(k + 1)
        q = zeros((k + 1, cnst.N0), dtype=complex)
        F0 = zeros((k + 1, cnst.N0), dtype=complex)

        for index in idx2:
            q[index, :] = exp(1j * 2 * pi * (f[index] - cnst.seq / cnst.N0))
            F0[index, :] = residualDFT0(q[index, :], cnst)

        AF0 = (A[idx2].T @ F0).reshape(-1)

        X0 = X - AF0.reshape(-1)

    for itit in range(cnst.Nya):
        ord = arange(cnst.K)
        for k in ord:
            pass
            idx3 = delete(ord, k)
            fdiff = f[idx3] - f[k]

            q05 = exp(1j * 2 * pi / cnst.N * (fdiff * cnst.N - 0.5))
            qm05 = exp(1j * 2 * pi / cnst.N * (fdiff * cnst.N + 0.5))
            F05 = residualDFT0(q05, cnst)
            Fm05 = residualDFT0(qm05, cnst)
            AF05_local = sum(A[idx3] * F05, axis=0)
            AFm05_local = sum(A[idx3] * Fm05, axis=0)

            Xti_05 = sum(x[0, :] * exp(-1j * 2 * pi * cnst.n / cnst.N * (f[k] * cnst.N + 0.5)))
            Xti_m05 = sum(x[0, :] * exp(-1j * 2 * pi * cnst.n / cnst.N * (f[k] * cnst.N - 0.5)))
            Xi_05 = Xti_05 - cnst.N * AF05_local
            Xi_m05 = Xti_m05 - cnst.N * AFm05_local

            z = cos(pi / cnst.N) - 1j * sin(pi / cnst.N) * (Xi_05 + Xi_m05) / (Xi_05 - Xi_m05)
            d = -1 / (2 * pi) * (log(z)).imag
            f[k] = f[k] + d

            fdiff = f[idx3] - f[k]
            q = exp(1j * 2 * pi * fdiff)
            F0 = residualDFT0(q, cnst)
            AF0_local = sum(A[idx3] * F0, axis=0)
            A[k] = 1 / cnst.N * (sum(x[0, :] * exp(-1j * 2 * pi * cnst.n * f[k]))) - AF0_local
    return A, f
