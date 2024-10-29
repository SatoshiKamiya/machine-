from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from csv_handler import CsvHandler

# モデルによる処理
class ModelProcess:
    # コンストラクタ
    def __init__(self, data_instance):
        print("ModelProcess__init__ data_instance:", data_instance)
        # 表示制限解除
        self.data_instance = data_instance

    

    

