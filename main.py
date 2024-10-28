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
    # 1つ目のモデル用
    csvHandler = CsvHandler("data/train.csv")

    # 全体のデータ取得　10レコードまで
    csvHandler.get_record(10)
    # データのtype確認
    csvHandler.get_column_type()
    # 行列数確認
    csvHandler.get_matrix_num()
    # 総データ数確認
    csvHandler.get_records_count()
    # カラム名取得
    csvHandler.get_column_label()
    # 欠損値確認
    csvHandler.get_data_isnull()

    # カラムはじき（）

    # ラベルエンコーディング

    # Age欠損値補完

    # データトレーニング前最終確認

    # ホールドアウト法によるトレーニングデータとテストデータ分割

    # モデルへ値を渡す


if __name__ == "__main__":
    main()
