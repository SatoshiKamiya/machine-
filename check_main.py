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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier

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


    X = train_df.drop(columns=target)
    y = train_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    models = {
        "LightGBM": LGBMClassifier(device='gpu', n_jobs=-1, verbose=-1),                      
        "XGBoost": XGBClassifier(eval_metric='logloss', tree_method='auto', verbosity=0)  
    }
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]

  #------------------------------ROC曲線による確認-----------------------------------
        # roc_auc = roc_auc_score(y_test, y_pred_proba)
        # print(f"{model_name} ROC AUC Score: {roc_auc:.4f}")
        
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        # plt.figure(figsize=(4, 3))
        # plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.4f})')
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
        # plt.legend(loc='lower right')
        # plt.show()
  #------------------------------交差検証---------------------------------------------

    for m_name, model in models.items():
        stratified_cv = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, X, y, cv=stratified_cv, scoring='roc_auc')
        print(f'------------{m_name}')
        print("Cross-validation scores:", cv_scores)
        print("Mean cross-validation score:", cv_scores.mean())

  #------------------------------アンサンブル学習-------------------------------------

    voting_clf = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')
    stratified_cv = StratifiedKFold(n_splits=10)
    cv_scores = cross_val_score(voting_clf, X, y, cv=stratified_cv, scoring='roc_auc')
    print(f'------------Voting Classifier')
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())


    voting_clf.fit(X, y)

    submission = pd.DataFrame(
        {"id": test_df["id"], "SalePrice": voting_clf.predict_proba(test)[:,1]}
    )
    
    # submission['loan_status'] = voting_clf.predict_proba(test)[:,1]
    
    submission.to_csv('submission.csv', index=False)




if __name__ == "__main__":
    main()
