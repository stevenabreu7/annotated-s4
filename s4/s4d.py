
from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from .s4 import (
    causal_convolution,
    cloneLayer,
    hippo_initializer,
    log_step_initializer,
    make_DPLR_HiPPO,
    scan_SSM,
)


if __name__ == '__main__':
    rng = jax.random.PRNGKey(1)


def discretize(A, B, step, mode="zoh"):
    if mode == "bilinear":
        num, denom = 1 + .5 * step*A, 1 - .5 * step*A
        return num / denom, step * B / denom
    elif mode == "zoh":
        return np.exp(step*A), (np.exp(step*A)-1)/A * B


def vandermonde_product(v, alpha, L):
    V = alpha[:, np.newaxis] ** np.arange(L)  # Vandermonde matrix
    return (v[np.newaxis, :] @ V)[0]


def s4d_kernel(C, A, L, step):
    Abar, Bbar = discretize(A, 1.0, step)
    return vandermonde_product(C * Bbar, Abar, L).real


@partial(jax.jit, static_argnums=2)
def s4d_kernel_zoh(C, A, L, step):
    kernel_l = lambda l: (C * (np.exp(step*A)-1)/A * np.exp(l*step*A)).sum()
    return jax.vmap(kernel_l)(np.arange(L)).real


def s4d_ssm(C, A, L, step):
    N = A.shape[0]
    Abar, Bbar = discretize(A, np.ones(N), step, mode="zoh")
    Abar = np.diag(Abar)
    Bbar = Bbar.reshape(N, 1)
    Cbar = C.reshape(1, N)
    return Abar, Bbar, Cbar


def test_conversion(N=8, L=16):
    """Test the equivalence of the S4D kernel with the generic SSM kernel."""
    step = 1.0 / L
    C = normal()(rng, (N, 2))
    C = C[..., 0] + 1j * C[..., 1]
    A, _, _, _ = make_DPLR_HiPPO(N)
    A = A[np.nonzero(A.imag > 0, size=N)]

    K_ = s4d_kernel(C, A, L, step)
    K = s4d_kernel_zoh(C, A, L, step)
    assert np.allclose(K_, K, atol=1e-4, rtol=1e-4)

    ssm = s4d_ssm(C, A, L, step)

    # # Apply CNN
    u = np.arange(L) * 1.0
    y1 = causal_convolution(u, K)

    # # Apply RNN
    _, y2 = scan_SSM(
        *ssm, u[:, np.newaxis], np.zeros((N,)).astype(np.complex64)
    )
    assert np.allclose(y1, y2.reshape(-1).real, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    test_conversion()


class S4DLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False
    scaling: str = "hippo"

    # The full training script has optimizer hooks that lower the LR on special params
    lr = {
        "A_re": 0.1,
        "A_im": 0.1,
        "B_re": 0.1,
        "B_im": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        if self.scaling == "hippo":
            init_A_re, init_A_im, _, _ = hippo_initializer(self.N)
            self.A_re = self.param("A_re", init_A_re, (self.N,))
            self.A_im = self.param("A_im", init_A_im, (self.N,))
        elif self.scaling == "linear":
            self.A_re = self.param("A_re", nn.initializers.constant(-0.5), (self.N,))
            def arange_initializer(scale):
                return lambda key, shape: scale * np.ones(shape) * np.arange(shape[-1])
            self.A_im = self.param("A_im", arange_initializer(np.pi), (self.N,))
        else: 
            raise NotImplementedError

        self.A = np.clip(self.A_re, None, -1e-4) + 1j * self.A_im
        self.B_re = self.param("B_re", nn.initializers.ones, (self.N,))
        self.B_im = self.param("B_im", nn.initializers.zeros, (self.N,))
        self.B = self.B_re + 1j * self.B_im
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            self.K = s4d_kernel_zoh(self.C, self.A, self.l_max, self.step)
        else:
            # FLAX code to ensure that we only compute discrete once during decoding.
            def init_discrete():
                return s4d_ssm(self.C, self.A, self.l_max, self.step)

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        if not self.decode:
            return causal_convolution(u, self.K) + self.D * u
        else:
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


S4DLayer = cloneLayer(S4DLayer)
