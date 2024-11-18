import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("main処理開始")

    pd.set_option("display.max_columns", None)  # 列の表示数の制限を解除
    pd.set_option("display.max_rows", None)  # 行の表示数の制限を解除
    # 1つ目のモデル用
    train_df = pd.read_csv("data/train.csv") 

    # 1つ目のモデル用
    test_df = pd.read_csv("data/test.csv") 

     # トレーニングデータとテストデータを合体させる
    # test_df = pd.read_csv("data/test.csv") 

    print("train_df　行列数")
    print(train_df.shape)  # (58645, 13)
    print("test_df　行列数")
    print(test_df.shape)  # (39098, 12)

    print("train_df　head")
    print(train_df.head(50)) 

    print("train_df　統計量")
    print(train_df.describe())

    print("train_df　欠損値")
    print(train_df.info())
    print("test_df　欠損値")
    print(test_df.info())

    print("相関係数チェック")
    corr_matrix = train_df.select_dtypes(include="number").corr() #　数値のみのカラム抽出
    print("相関係数チェック　ターゲット")
    print(corr_matrix["loan_status"])
    print("相関係数チェック　全体")
    print(corr_matrix)

    print("値の分布チェック")

    print("train_df　ターゲット値可視化")
    train_df.hist(figsize=(10, 10), bins=30)
    plt.show()

    

if __name__ == "__main__":
    main()
