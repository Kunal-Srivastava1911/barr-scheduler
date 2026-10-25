"""Build schedulers from short names, e.g. "RR-5", "BARR-GBRT", "BARR-S-GBRT".

Every call returns a fresh scheduler (fresh predictor state), so the same name
can be run on many workloads.
"""

from ..predictors.base import OraclePredictor
from ..predictors.ema import EMAPredictor
from .barr import BARR
from .dyn_mean import MeanRR
from .dyn_median import MedianRR
from .fcfs import FCFS
from .rr import RoundRobin
from .sjf import SJF, SRTF

# Table III of the implementation plan, plus BARR-S from the final review
ALL = ["FCFS", "RR-2", "RR-5", "RR-10", "RR-20", "RR-50", "SJF-EMA", "SRTF",
       "MedianRR-oracle", "MedianRR-pred", "BeheraRR-oracle", "BeheraRR-pred",
       "MeanRR-oracle", "MeanRR-pred",
       "BARR-EMA", "BARR-GBRT", "BARR-ORACLE", "BARR-S-EMA", "BARR-S-GBRT", "BARR-S-ORACLE"]

ORACLE = {"SRTF", "MedianRR-oracle", "BeheraRR-oracle", "MeanRR-oracle",
          "BARR-ORACLE", "BARR-S-ORACLE"}

DEFAULT_BARR = {"delta": 0.25, "q_min": 1.0, "q_max": 50.0, "beta": 2.0, "aging": 64}
DEFAULT_BARR_S = {"delta": 0.25, "q_min": 1.0, "q_max": 5.0, "beta": 2.0, "aging": 64}


def label(name):
    return {"SRTF": "SRTF oracle"}.get(name, name.replace("RR-", "RR q=", 1) if name.startswith("RR-") else name)


def _predictor(tag, trained):
    tag = tag.lower()
    if tag == "oracle":
        return OraclePredictor()
    if trained and tag in trained:
        return trained[tag].fresh()
    if tag == "ema":
        return EMAPredictor()
    raise KeyError(f"predictor {tag!r} is not trained (run python -m barr.predictors.train)")


def make(name, trained=None, barr=None, barr_s=None):
    """trained: dict of trained predictors (barr.predictors.train)."""
    ema = (trained or {}).get("ema")
    ema = ema.fresh() if ema else EMAPredictor()
    if name == "FCFS":
        return FCFS()
    if name.startswith("RR-"):
        return RoundRobin(float(name[3:]))
    if name == "SJF-EMA":
        return SJF(ema)
    if name == "SJF-ORACLE":
        return SJF(OraclePredictor(), name="SJF oracle")
    if name == "SRTF":
        return SRTF()
    if name.startswith(("MedianRR-", "BeheraRR-", "MeanRR-")):
        base, mode = name.split("-")
        mode = "oracle" if mode == "oracle" else "pred"
        if base == "MeanRR":
            return MeanRR(mode, ema)
        return MedianRR(mode, ema, reorder=base == "BeheraRR")
    if name.startswith("BARR-S-"):
        cfg = {**DEFAULT_BARR_S, **(barr_s or {})}
        return BARR(_predictor(name[7:], trained), order="spf", name=name, **cfg)
    if name.startswith("BARR-"):
        cfg = {**DEFAULT_BARR, **(barr or {})}
        return BARR(_predictor(name[5:], trained), order="fifo", name=name, **cfg)
    raise KeyError(f"unknown scheduler {name!r}")
