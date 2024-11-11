import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def main():
    print("main処理開始")
    # 1つ目のモデル用
    train_df = pd.read_csv("data/customer_transaction_prediction/train.csv") # (200000, 202)
    train_df_first = train_df.iloc[:,:100] # メモリ不足のため分割（前半）
    train_df_second = train_df.iloc[:,100:150] # メモリ不足のため分割（中）
    train_df_third = train_df.iloc[:,150:] # メモリ不足のため分割（後半）
    test_df = pd.read_csv("data/customer_transaction_prediction/test.csv") # (200000, 201)
    test_df_first = test_df.iloc[:,:100] # メモリ不足のため分割（前半）
    test_df_second = test_df.iloc[:,100:150] # メモリ不足のため分割（中）
    test_df_third = test_df.iloc[:,150:] # メモリ不足のため分割（後半）

    total_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    print("tranigデータ数：", train_df.shape) # 1460
    print("testデータ数：", test_df.shape) # 1459

    # # 列数50まで表示
    # pd.set_option("display.max_columns", 50)

    # 欠損値チェック # なし
    # print(train_df_first.info()) #なし
    

    # ---------------------------カラム相関チェック後抽出後データ--------------------------------


if __name__ == "__main__":
    main()
