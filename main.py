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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


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

    print("----------------- data divide --------------------")
    # ホールドアウト法によるトレーニングデータとテストデータ分割
    # データセットを trainとtestに分割
    df = csvHandler.get_data_instance()
    train = df[df["Survived"].notnull()]
    test = df[df["Survived"].isnull()].drop("Survived", axis=1)

    # データフレームをnumpyに変換
    X = train.values[:, 1:]  # トレーニングデータの1行目以降取得
    y = train.values[:, 0]  # トレーニングデータの0行目のみ取得
    # test_x = test.values

    # print("----------------- gred serch start --------------------")
    # # グリッドサーチのためのデータ分割
    # # test_size:テストデータの割合　0.2=20%
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # # モデルのインスタンスを作成
    # model = RandomForestClassifier(random_state=10, warm_start=True)
    # # 試したいパラメータのリスト
    # param_grid = {
    #     "n_estimators": [15, 25, 50, 100, 150, 300, 450],
    #     "max_depth": [3, 5, 7, 8, 9, 10, 15, 20, 25],
    # }

    # # グリッドサーチ
    # grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    # grid_search.fit(X_train, y_train)

    # # 最適なパラメータとスコアの表示
    # print("Best parameters:", grid_search.best_params_)
    # print("Best cross-validation score:", grid_search.best_score_)

    # # テストデータに対する精度
    # best_model = grid_search.best_estimator_
    # y_pred = best_model.predict(X_test)
    # print("Test accuracy:", accuracy_score(y_test, y_pred))

    # # Best parameters: {'max_depth': 8, 'n_estimators': 300}
    # # Best cross-validation score: 0.8370432384516893
    # # Test accuracy: 0.8212290502793296
    # print("----------------- gred serch end --------------------")

    # ----------------------------------------------------------------------------------------
    print("----------------- trainig start --------------------")
    # ランダムフォレスト（2値分類）　ハイパーパラメータが決まったらこれを利用する
    clf = RandomForestClassifier(
        random_state=10,  # ランダムで得られる数値（数値を指定すれば固定される）
        warm_start=True,  # 既にフィットしたモデルに学習を追加
        n_estimators=300,  # 使用する決定木の数
        max_depth=8,  # 各決定木の深さの最大値
        max_features="sqrt",  # 各決定木で使用する特徴量の数
    )

    pipeline = make_pipeline(clf)
    pipeline.fit(X, y)

    # フィット結果の表示
    cv_result = cross_validate(pipeline, X, y, cv=10)
    print(
        "mean_score = ", np.mean(cv_result["test_score"])
    )  # 結果：mean_score = 0.8440324594257177
    print(
        "mean_std = ", np.std(cv_result["test_score"])
    )  # 結果：mean_std = 0.04348198471288303


if __name__ == "__main__":
    main()
