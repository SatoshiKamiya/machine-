import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    print("main処理開始")
    # 1つ目のモデル用
    train_df = pd.read_csv("data/house_prices/train.csv")
    test_df = pd.read_csv("data/house_prices/test.csv")
    total_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    # レコード数
    print("records count=", total_df.shape[0])
    # 型＆nullチェック
    print("info=", total_df.info())
    # データ一部表示
    print("head")
    print(total_df.head())

    # 相関係数
    numeric_df = train_df.select_dtypes(include=[float, int])
    # カラムチェック
    print("カラムの種類", numeric_df.columns)

    cor_train_df_01 = train_df[
        [
            "SalePrice",
            "LotFrontage",
            "OverallQual",
            "YearBuilt",
            "YearRemodAdd",
            "MasVnrArea",
            "BsmtFinSF1",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "GrLivArea",
            "FullBath",
        ]
    ]

    # cor_train_df_02 = train_df[
    #     [
    #         "SalePrice",
    #         "TotRmsAbvGrd",
    #         "Fireplaces",
    #         "GarageYrBlt",
    #         "GarageCars",
    #         "GarageArea",
    #         "WoodDeckSF",
    #         "OpenPorchSF",
    #     ]
    # ]

    cor = cor_train_df_01.corr()
    sns.heatmap(
        cor,
        cmap=sns.color_palette("coolwarm", 10),
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )
    plt.show()
    # print(correlation_matrix)


if __name__ == "__main__":
    main()
