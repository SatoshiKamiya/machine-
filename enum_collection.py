from enum import Enum


# 端数処理
class Rounding(Enum):
    NON = 0
    ROUNDINGUP = 1 # 四捨五入
    TRUNCATE = 2 # 切り捨て