import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
def main():
    print("main処理開始")

    pd.set_option("display.max_columns", None)  # 列の表示数の制限を解除
    pd.set_option("display.max_rows", None)  # 行の表示数の制限を解除
    # 1つ目のモデル用
    train_df = pd.read_csv("data/train.csv") 

    # 1つ目のモデル用
    test_df = pd.read_csv("data/test.csv") 

    target = "loan_status"

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
    # train_df.hist(figsize=(10, 10), bins=30)
    # plt.show()
    number_columns = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
    # train_num_df = train_df[number_columns]
    train_num_df = train_df.select_dtypes(include="number")
    sns.pairplot(train_num_df)  
    plt.show() 
    plt.close()  

    #------------------------------labael encoding-----------------------------------
    obj_columns = ["loan_status", "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    train_obj_df = train_df[obj_columns]
    for column in train_obj_df.select_dtypes(include=['object']).columns:
        train_obj_df[column] = LabelEncoder().fit_transform(train_obj_df[column])

    print("train_obj_df 相関係数チェック")
    corr_matrix = train_obj_df.select_dtypes(include="number").corr() #　数値のみのカラム抽出
    print("相関係数チェック　ターゲット")
    print(corr_matrix["loan_status"])
    print("相関係数チェック　全体")
    print(corr_matrix)

    

    #------------------------------one hot encoding-----------------------------------
    # obj_columns = ["loan_status", "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    # train_obj_df = train_df[obj_columns]


    # # 各カラムの値と出現回数を取得
    # print("各カラムの値と出現回数を取得")
    # for column in train_obj_df.columns:
    #     print(f"Column: {column}")
    #     print(train_obj_df[column].value_counts())
    #     print("\n")

    
    # # One-Hot Encodingを適用
    # df_encoded = pd.get_dummies(train_obj_df).astype(int)
    # print("One-Hot Encodingを適用")
    # print(df_encoded.head(10))

    # correlation_matrix = df_encoded.corr()
    # correlation_with_target = correlation_matrix["loan_status"].sort_values(ascending=False)

    # # ガレージの外装仕上げカテゴリのみの相関を表示
    # correlation_with_target[df_encoded.columns.str.startswith('person_home_ownership_')]
    # print("loan_status vs person_home_ownership")
    # print(correlation_with_target)




if __name__ == "__main__":
    main()
