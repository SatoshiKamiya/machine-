import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from csv_handler import CsvHandler
from model_process import ModelProcess
from enum_collection import Rounding

def main():
    print("main処理開始")
    csvHandler = CsvHandler("data/train.csv")

    # データチェック
    csvHandler.get_table_info()

    # 指定したデータチェック
    csvHandler.get_specification_record('Age', display_num=891)


    # カラムを指定して欠損値のあるレコードNoを取得
    records_number_array = csvHandler.get_drop_records_number('Age')
    
    # ランダムフォレストによる補完
    csvHandler.random_forest(['Age', 'Pclass','Sex','Parch','SibSp'])

    # 補完の値チェック
    csvHandler.get_assignment_records('Age', records_number_array)

    # 端数処理（カラム指定）
    csvHandler.rounding_process(['Age'],  Rounding.ROUNDINGUP, 5)


    # 値の統計量
    # csvHandler.random_forest(['Age', 'Pclass','Sex','Parch','SibSp'])





    
    # csvHandler.drop_records([2,6,8])
    # csvHandler.drop_records_area(2,8)
    # count = csvHandler.get_records_count()
    # csvHandler.get_specification_record(['Age'], count)
    # csvHandler.average_value_interpolation('Age')
    # csvHandler.mode_value_interpolation('Age') # ←average_value_interpolationと組み合わせるとうまく起動しない
    # csvHandler.average_value('Age')
    # csvHandler.median_value('Age')
    # csvHandler.mode_value('Age')
    # csvHandler.correlation_coefficient_oto('Age')
    # csvHandler.correlation_coefficient_otm()
    # csvHandler.multiple_imputation_MICE(['Pclass','Age','SibSp'])

    # csvHandler.show_hist()
    # csvHandler.show_kds()
    # csvHandler.show_pair_plot()

if __name__ == "__main__":
    main()
