import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def main():
    print("main処理開始")
    # 1つ目のモデル用
    train_df = pd.read_csv("data/house_prices/train.csv")
    test_df = pd.read_csv("data/house_prices/test.csv")
    total_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    # ---------------------------カラム相関チェック後抽出後データ--------------------------------
    total_df = total_df[
        [
            "SalePrice",
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
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
        ]
    ]

    # 文字列は別途用意　まずは数値のみデータの外れ値と欠損値補完する

    # -----------------------------------------------------------------------------------------

    # レコード数
    print("records count=", total_df.shape[0])
    # 型＆nullチェック
    print("info=", total_df.info())
    # データ一部表示
    print("head")
    print(total_df.head())

    MasVnrArea_df = total_df[["MasVnrArea"]]
    # ---------------------------各データの状態チェック--------------------------------
    # ヒストグラム
    # MasVnrArea_df.hist(figsize=(10, 10), bins=30)
    # plt.show()
    # 箱ひげ図
    # plt.figure(figsize=(8, 6))
    # sns.boxplot(total_df["MasVnrArea"])
    # plt.ylabel("Values")
    # plt.show()
    # 欠損値のある行番号を取得
    result = total_df[total_df["MasVnrArea"].isnull()].index
    print(result)

    complement_MasVnrArea_df = total_df[
        [
            "OverallQual",
            "MasVnrArea",
            "TotalBsmtSF",
            "1stFlrSF",
            "GrLivArea",
            "GarageCars",
            "GarageArea",
        ]
    ]

    # 相関係数チェック
    df_corr = complement_MasVnrArea_df.corr()
    print(df_corr)
    print(type(df_corr))

    # ランダムフォレストによる補完
    print("ランダムフォレストによる補完開始")

    # 学習データとテストデータに分離し、numpyに変換
    known_target_df = complement_MasVnrArea_df.notnull().values
    unknown_target_df = complement_MasVnrArea_df.isnull().values

    # 学習データをX, yに分離
    X = known_target_df[:, 1:]
    y = known_target_df[:, 0]

    # ランダムフォレストで推定モデルを構築
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    rfr.fit(X, y)

    # 推定モデルを使って、テストデータのAgeを予測し、補完
    predictedAges = rfr.predict(unknown_target_df[:, 1::])
    complement_MasVnrArea_df.loc[(complement_MasVnrArea_df.isnull()), 'MasVnrArea'] = (
        predictedAges
    )


# -----------------------------------------------------------------------------------------
# # 相関係数
# numeric_df = train_df.select_dtypes(include=[float, int])
# # カラムチェック
# print("カラムの種類", numeric_df.columns)

# salePrices check
# print("セール価格の統計値")
# print(train_df["SalePrice"].describe())

# ---------------------------One-Hot Encoding相関関係チェック--------------------------------
# print("Neighborhood")
# print(train_df["Neighborhood"].value_counts())
# print("ExterQual")
# print(train_df["ExterQual"].value_counts())
# print("BsmtQual")
# print(train_df["BsmtQual"].value_counts())
# print("KitchenQual")
# print(train_df["KitchenQual"].value_counts())
# print("GarageFinish")
# print(train_df["GarageFinish"].value_counts())

# train_df = train_df[["SalePrice", "ExterQual", "BsmtQual", "BsmtQual"]]

# One-Hot Encodingを適用
# df_encoded = pd.get_dummies(train_df, columns=["ExterQual"]).astype(int)
# df_encoded = pd.get_dummies(train_df).astype(int)

# print("中間チェック")
# print(df_encoded)
# 相関を計算
# bsmtqual_columns = [
#     col for col in df_encoded.columns if col.startswith("ExterQual" + "_")
# ]
# correlation_matrix = df_encoded.corr()["SalePrice"][bsmtqual_columns]

# print("SalePrice vs GarageFinish")
# print(correlation_matrix)

# ---------------------------文字列データ選定--------------------------------
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
