import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def main():
    print("main処理開始")

    target = 'loan_status'

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

    

    class DataCleaner:
        def __init__(self):
            self.ti = {}
            self.ki = KNNImputer()
    
        def fit_label_encoders(self, df):
            cat_features = df.select_dtypes(include=["object"]).columns
            for feature in cat_features:
                le = LabelEncoder()
                le.fit(df[feature])
                self.ti[feature] = le
    
        def transform_labels(self, df):
            for feature, le in self.ti.items():
                df[feature] = df[feature].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            return df
    
        def impute_numeric(self, df):
            missed_numeric_features = df.select_dtypes(include=[float, int]).columns[df.isna().mean() > 0]
            if not missed_numeric_features.empty:
                df[missed_numeric_features] = self.ki.fit_transform(df[missed_numeric_features])
            else:
                print("No numeric columns with missing values to impute.")
            return df
    
        def clean_data(self, df, train):
            if train:
                self.fit_label_encoders(df)
            df = self.transform_labels(df)
            return self.impute_numeric(df)
    
    data_cleaner = DataCleaner()
    train_df = data_cleaner.clean_data(train_df, train=True)
    test_df = data_cleaner.clean_data(test_df, train=False)
    print("train_df head after cleaner")
    print(train_df.head(50))
    print("test_df head after cleaner")
    print(test_df.info(50))


    def create_features(train, test):
        for df in [train, test]:
            df['income_to_age'] = df['person_income'] / df['person_age']
            df['loan_to_income'] = df['loan_amnt'] / df['person_income']
            df['rate_to_loan'] = df['loan_int_rate'] / df['loan_amnt']
            df['age_squared'] = df['person_age'] ** 2
            df['log_income'] = np.log1p(df['person_income'])
            df['age_credit_history_interaction'] = df['person_age'] * df['cb_person_cred_hist_length']
            df['high_loan_to_income'] = (df['loan_percent_income'] > 0.5).astype(int)
            df['loan_to_employment'] = df['loan_amnt'] / (df['person_emp_length'] + 1)
            df['is_new_credit_user'] = (df['cb_person_cred_hist_length'] < 2).astype(int)
            df['rate_to_grade'] = df.groupby('loan_grade')['loan_int_rate'].transform('mean')
            df['high_interest_rate'] = (df['loan_int_rate'] > df['loan_int_rate'].mean()).astype(int)
            df['age_to_credit_history'] = df['person_age'] / (df['cb_person_cred_hist_length'] + 1)
            df['income_home_mismatch'] = ((df['person_income'] > df['person_income'].quantile(0.8)) & (df['person_home_ownership'] == 'RENT')).astype(int)
            df['normalized_loan_amount'] = df.groupby('loan_intent')['loan_amnt'].transform(lambda x: (x - x.mean()) / x.std())
            df['income_to_loan'] = df['person_income'] / df['loan_amnt']
            df['age_cubed'] = df['person_age'] ** 3
            df['log_loan_amnt'] = np.log1p(df['loan_amnt'])
            df['age_interest_interaction'] = df['person_age'] * df['loan_int_rate']
            df['credit_history_to_age'] = df['cb_person_cred_hist_length'] / df['person_age']
            df['high_loan_amount'] = (df['loan_amnt'] > df['loan_amnt'].quantile(0.75)).astype(int)
            df['rate_to_credit_history'] = df['loan_int_rate'] / (df['cb_person_cred_hist_length'] + 1)
            df['intent_home_match'] = ((df['loan_intent'] == 'HOMEIMPROVEMENT') & (df['person_home_ownership'] == 'OWN')).astype(int)
            df['creditworthiness_score'] = (df['person_income'] / (df['loan_amnt'] * df['loan_int_rate'])) * (df['cb_person_cred_hist_length'] + 1)
            df['age_to_employment'] = df['person_age'] / (df['person_emp_length'] + 1)
            df['age_income_mismatch'] = ((df['person_age'] < 30) & (df['person_income'] > df['person_income'].quantile(0.9))).astype(int)
            df['rate_to_age'] = df['loan_int_rate'] / df['person_age']
            df['high_risk_flag'] = ((df['loan_percent_income'] > 0.4) &
                                    (df['loan_int_rate'] > df['loan_int_rate'].mean()) &
                                    (df['cb_person_default_on_file'] == 'Y')).astype(int)
    
            df['age_sin'] = np.sin(2 * np.pi * df['person_age'] / 100)
            df['age_cos'] = np.cos(2 * np.pi * df['person_age'] / 100)
            df['stability_score'] = (df['person_emp_length'] * df['person_income']) / (df['loan_amnt'] * (df['cb_person_cred_hist_length'] + 1))
    
        return train, test
    
    train_df, test = create_features(train_df, test_df)
    print("create_features after result")
    # print(train_df.shape)
    # plt.figure(figsize=(25,15))
    # sns.heatmap(train_df.corr(method='spearman'), annot=True, fmt=".1f", annot_kws={"size": 8})
    # plt.show()

    X = train_df.drop(columns=target)
    y = train_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    models = {
        "LightGBM": LGBMClassifier(device='gpu', n_jobs=-1, verbose=-1),                      
        "XGBoost": XGBClassifier(eval_metric='logloss', tree_method='gpu_hist', verbosity=0)  
    }
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"{model_name} ROC AUC Score: {roc_auc:.4f}")
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()



  #------------------------------ここから上はサイトからすべて引用-----------------------------------

    # print("相関係数チェック")
    # corr_matrix = train_df.select_dtypes(include="number").corr() #　数値のみのカラム抽出
    # print("相関係数チェック　ターゲット")
    # print(corr_matrix["loan_status"])
    # print("相関係数チェック　全体")
    # print(corr_matrix)


    # print("train_df　ターゲット値可視化")
    # # train_df.hist(figsize=(10, 10), bins=30)
    # # plt.show()
    # # number_columns = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
    # number_columns = ["person_age", "loan_percent_income", "cb_person_cred_hist_length"]
    # # train_num_df = train_df[number_columns]
    # sns.pairplot(train_df[number_columns], diag_kind="kde")  # diag_kind="kde"で対角線にカーネル密度推定を表示
    # plt.show()
    #------------------------------labael encoding-----------------------------------
    # obj_columns = ["loan_status", "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    # train_obj_df = train_df[obj_columns]
    # for column in train_obj_df.select_dtypes(include=['object']).columns:
    #     train_obj_df[column] = LabelEncoder().fit_transform(train_obj_df[column])

    # print("train_obj_df 相関係数チェック")
    # corr_matrix = train_obj_df.select_dtypes(include="number").corr() #　数値のみのカラム抽出
    # print("相関係数チェック　ターゲット")
    # print(corr_matrix["loan_status"])
    # print("相関係数チェック　全体")
    # print(corr_matrix)

    

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
