import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def main():
    print("main処理開始")
    # 1つ目のモデル用
    train_df = pd.read_csv("data/house_prices/train.csv")
    test_df = pd.read_csv("data/house_prices/test.csv")
    total_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    # print("tranigデータ数：", train_df.shape[0]) # 1460
    # print("testデータ数：", test_df.shape[0]) # 1459

    # 列数50まで表示
    pd.set_option("display.max_columns", 50)

    # ---------------------------カラム相関チェック後抽出後データ--------------------------------


if __name__ == "__main__":
    main()
