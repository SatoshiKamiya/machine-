import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("main処理開始")

    pd.set_option("display.max_columns", None)  # 列の表示数の制限を解除
    pd.set_option("display.max_rows", None)  # 行の表示数の制限を解除
    # 1つ目のモデル用
    train_df = pd.read_csv(
        "data/train.csv"
    )  # (200000, 202)

        # 1つ目のモデル用
    test_df = pd.read_csv(
        "data/test.csv"
    )  # (200000, 202)

    print("train_df　行列数")
    print(train_df.shape) 
    print("test_df　行列数")
    print(test_df.shape) 

    print("train_df　head")
    print(train_df.head()) 

    print("train_df　統計量")
    print(train_df.describe())

    print("train_df　欠損値")
    print(train_df.info())
    print("test_df　欠損値")
    print(test_df.info())

    print("train_df　ターゲット値可視化")
    train_df.hist(figsize=(10, 10), bins=30)
    plt.show()

if __name__ == "__main__":
    main()
