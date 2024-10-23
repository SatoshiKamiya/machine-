from enum import Enum


# 端数処理
class Rounding(Enum):
    NON = 0
    ROUNDINGUP = 1
    TRUNCATE = 2