import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def trained():
    """A small GBRT predictor, trained in a second, for policies that need one."""
    from barr.features import build_dataset, to_log
    from barr.predictors import GBRT, LearnedPredictor
    from barr.predictors.train import training_procs

    procs = training_procs(n_procs=20, bursts=(20, 40), include_linux=False)
    d = build_dataset(procs)
    m = GBRT(n_trees=30, depth=4).fit(d["train"][0], to_log(d["train"][1]),
                                      d["val"][0], to_log(d["val"][1]))
    return {"gbrt": LearnedPredictor(m, "gbrt")}
