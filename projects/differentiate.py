import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[0] = 1
    C = (1 - dt)/(1 + dt)
    for n in range(0, len(out)-1):
        out[n+1] = C* (out[n] if n>0 else out[0]) 
    return out

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(u)
    out[0] = 1
    out[1:] = (1 - dt)/(1 + dt)
    out[:] = np.cumprod(out)
    return out

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()