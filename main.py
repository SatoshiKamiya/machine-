import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from csv_handler import CsvHandler
from model_process import ModelProcess
from enum_collection import Rounding
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate


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

    # カラムはじき（PassengerId Name SibSp Parch Ticket Cabin Embarked）
    # 必要なカラム：Survived　Pclass　Sex　Age　Fare　←　本来新たな特徴量を作成する特徴量エンジニアリングをする必要あり
    csvHandler.select_columns_data(["Survived", "Pclass", "Sex", "Age", "Fare"])

    # ラベルエンコーディング
    csvHandler.label_encoder(["Sex"])

    # Ageの欠損値レコード
    records_number_array = csvHandler.get_drop_records_number("Age")

    # Age欠損値補完
    csvHandler.random_forest("Age", ["Age", "Pclass", "Sex"])

    # ヒストグラムでチェック
    # csvHandler.show_part_column_recrods_hist("Age", records_number_array)

    # Age単数処理
    csvHandler.rounding_process("Age", Rounding.TRUNCATE, 0)

    # データトレーニング前最終確認
    csvHandler.get_record(10)
    csvHandler.get_data_isnull()

    # ホールドアウト法によるトレーニングデータとテストデータ分割
    # データセットを trainとtestに分割
    df = csvHandler.get_data_instance()
    train = df[df["Survived"].notnull()]
    test = df[df["Survived"].isnull()].drop("Survived", axis=1)

    # データフレームをnumpyに変換
    X = train.values[:, 1:]
    y = train.values[:, 0]
    test_x = test.values

    # モデルへ値を渡す
    # 採用する特徴量を25個から20個に絞り込む
    select = SelectKBest(k=20)

    clf = RandomForestClassifier(
        random_state=10,
        warm_start=True,  # 既にフィットしたモデルに学習を追加
        n_estimators=26,
        max_depth=6,
        max_features="sqrt",
    )
    pipeline = make_pipeline(select, clf)
    pipeline.fit(X, y)

    # フィット結果の表示
    cv_result = cross_validate(pipeline, X, y, cv=10)
    print("mean_score = ", np.mean(cv_result["test_score"]))
    print("mean_std = ", np.std(cv_result["test_score"]))


if __name__ == "__main__":
    main()
