import numpy as np

from quicc_dynavis.comparison import resolve_combined_time_limits

def test_default_limits():
    limits = resolve_combined_time_limits(
        [
            np.array([1.0, 2.0]),
            np.array([0.0, 4.0]),
        ]
    )

    assert limits == (-0.2, 4.2)


def test_user_xlim():
    limits = resolve_combined_time_limits(
        [np.array([1.0, 2.0])],
        xlim=(10, 20),
    )

    assert limits == (10, 20)
