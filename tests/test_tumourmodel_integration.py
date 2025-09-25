import numpy as np
import tumourmodel as tm


def test_simulate_smoke():
    print("Running smoke test for tumourmodel.simulate()")
    theta = [0.05, 0.01, 30, 48, 5]  # p0, psc, dmax, gage, page
    T = 5
    y = tm.simulate(theta, T=T, seed=123, start_volume=50.0)
    print("Simulated data:", y)
    assert isinstance(y, np.ndarray)
    assert y.shape == (T,)
    print("Passed smoke test for tumourmodel.simulate()")
