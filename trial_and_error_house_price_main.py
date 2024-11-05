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

    # salePrices check
    print("セール価格の統計値")
    print(train_df["SalePrice"].describe())

    # `---------------------------文字列データ選定--------------------------------
    # cor_str_train_df = train_df[
    #     [
    #         "Neighborhood",
    #         "ExterQual",
    #         "BsmtQual",
    #         "KitchenQual",
    #         "GarageFinish",
    #     ]
    # ]
    # print("info=", cor_str_train_df.info())
    # `---------------------------文字列内の相関選定--------------------------------
    # cor_str_train_df_01 = train_df[
    #     [
    #         "SalePrice",
    #         "Street",
    #         "Alley",
    #         "LotShape",
    #         "LandContour",
    #         "Utilities",
    #         "LotConfig",
    #         "LandSlope",
    #         "Neighborhood",
    #         "Condition1",
    #         "Condition2",
    #         "BldgType",
    #         "HouseStyle",
    #     ]
    # ]

    # pd.set_option("display.max_rows", 500)
    # print("head")
    # print(cor_str_train_df_01.head(300))

    # `---------------------------相関係数による相関選定----------------------------
    # cor_train_df_01 = train_df[
    #     [
    #         "SalePrice",
    #         "LotFrontage",
    #         "OverallQual",
    #         "YearBuilt",
    #         "YearRemodAdd",
    #         "MasVnrArea",
    #         "BsmtFinSF1",
    #         "TotalBsmtSF",
    #         "1stFlrSF",
    #         "2ndFlrSF",
    #         "GrLivArea",
    #         "FullBath",
    #     ]
    # ]

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

    # cor_train_df_03 = train_df[
    #     [
    #         "SalePrice",
    #         "BedroomAbvGr",
    #         "KitchenAbvGr",
    #     ]
    # ]

    # cor = cor_train_df_03.corr()
    # sns.heatmap(
    #     cor,
    #     cmap=sns.color_palette("coolwarm", 10),
    #     annot=True,
    #     fmt=".2f",
    #     vmin=-1,
    #     vmax=1,
    # )
    # plt.show()
    # print(correlation_matrix)


if __name__ == "__main__":
    main()
