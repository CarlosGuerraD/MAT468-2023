import numpy as np
from scipy import stats
import pickle
import datetime
from multiprocessing import Pool

unif = np.random.default_rng().uniform
normal = np.random.default_rng().normal

def Normal_pp(p, sigma=1):
    A = np.array(normal(0, sigma, p**2)).reshape((p,p))
    return A


def Wishart_pp(p, sigma=1):
    A = np.array(normal(0, sigma, p**2)).reshape((p,p))
    return A.T @ A


def gamma_p(k, p):
    product = np.pi**(p*(p-1)/4)

    for i in range(p):
        product *= stats.gamma((k/2) - (i-1)/2).pdf(1)

    return product


def f(X, k, V, invV):
    min_eigval = min(np.linalg.eigvals(X))

    if min_eigval< 1e-10:
        return 0
    
    p = V.shape[0]
    detX = np.linalg.det(X)
    detV = np.linalg.det(V)
    trace = np.trace(invV @ X)
    num = np.exp(-trace/2) * detX**((k-p-1)/2)
    denom = 2**(k*p/2) * (detV**(k/2)) * gamma_p(k,p)
    return num/denom


def reduced_f(X, k, V, invV):
    min_eigval = min(np.linalg.eigvals(X))

    if min_eigval < 1e-10:
        return 0
    
    p = V.shape[0]
    detX = np.linalg.det(X)
    trace = np.trace(invV@X)
    _ = np.exp(-trace/2) * (detX**((k-p-1)/2))
    return _

    
def simulate_Wishart(args):
    p = args["p"]
    r = args["r"]
    sigma_V = args["sigma_V"]
    sigma_Yt = args["sigma_Yt"]
    n = args["n"]

    print(f"Simulando {p}x{p}")

    start = datetime.datetime.now()

    k = p + r
    I = np.identity(p)
    V = Wishart_pp(p, 1) + np.sqrt(p) * I
    invV = np.linalg.inv(V)

    X_chain = np.zeros((n, p, p))
    X_chain[0] = Wishart_pp(p, sigma_Yt * 0.9**p) + p**2 * I

    for i in range(n - 1):
        Xt = X_chain[i]
        Zt = Normal_pp(p, sigma_Yt)
        Yt = Xt + (1/2) * (Zt + Zt.T + I)
        Ut = unif(0,1)
        a = np.min([1, reduced_f(Yt, k, V, invV) / reduced_f(Xt, k, V, invV)])

        if Ut < a:
            X_chain[i + 1] = Yt
        else:
            X_chain[i + 1] = Xt

    end = datetime.datetime.now()
    file = open(f"./dataP1/file_{str(p).zfill(2)}.obj", "wb")
    pickle.dump((V, X_chain, (end - start).seconds), file)
    file.close()
    print(f"Terminado {p}x{p}")

if __name__ == "__main__":

    sigma_V = 2
    sigma_Yt = 10
    n = 1_000_000
    r = 3.5

    with Pool(processes = 16) as pool:
        pool.map(simulate_Wishart, [{
            "p" : i,
            "r" : r,
            "sigma_Yt" : sigma_Yt,
            "sigma_V" : sigma_V,
            "n" : n
        } for i in range(2, 11)])