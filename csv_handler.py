import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.imputation.mice import MICEData
from sklearn.ensemble import RandomForestRegressor

# csvファイル処理クラス
class CsvHandler:
    # コンストラクタ
    def __init__(self, path):
        print("CsvHandler__init__ path:", path)
        # 表示制限解除
        pd.set_option('display.max_rows', None)
        
        # csvファイル保存処理
        self.path = path

        if os.path.exists(path):
            # オリジナルデータ変数
            self._original_csv_data =  pd.read_csv(path)
              # 変更に対応するデータ変数
            self._csv_data = self._original_csv_data
        else:
            print(f"ファイルが見つかりません: {path}")


#データ取得 
    # インスタンス
    def get_original_data_instance(self):
        return self._original_csv_data
    
    # 1つのカラム取得
    def choose_column_data_instance(self, column_name):
        return self._original_csv_data[column_name]
    
    # 複数カラム取得
    def choose_columns_data_instance(self, column_names):
        return self._original_csv_data[column_names]


#データチェック
    # データ取得
    def get_record(self, record_count):
        print("CsvHandler get_record")
        result = self._csv_data.head(record_count)
        print(result)
    
    # データ取得（カラム指定）
    def get_specification_record(self, labels, display_num=10):
        print("CsvHandler get_record")
        result = self._csv_data[labels].head(display_num)
        print(result)

    # データチェック
    def get_table_info(self):
        print("CsvHandler get_table_info")
        result = self._csv_data.info
        print(result)

    # カラム表示
    def get_column_label(self):
        print("CsvHandler get_column_label")
        result = self._csv_data.columns
        print(result)

    # 行列数
    def get_matrix_num(self):
        print("CsvHandler get_matrix_num")
        result = self._csv_data.shape
        print(result)

    # 行列数
    def get_matrix_num(self):
        print("CsvHandler get_matrix_num")
        result = self._csv_data.shape
        print(result)

    # 総データ数取得チェック
    def get_records_count(self):
        result = self._csv_data.shape[0]
        print("CsvHandler get_records_count:", result)
        return result

     # 各カラムの欠損値数チェック
    def get_data_describe(self):
        print("CsvHandler get_data_describe")
        result = self._csv_data.describe()
        print(result)   

    # path取得
    def get_Path(self):
        return self.path

    # データ初期化
    def reset_data(self):
        self._csv_data = self._original_csv_data
    
    # カラムを指定して欠損値のある行を指定
    def get_drop_records_number(self, column):
        result = self._csv_data[self._csv_data[column].isnull()].index
        print("CsvHandler get_drop_records_number result:", result)
        return result

    # 行数を指定してそのレコードを取得
    def get_assignment_records(self, column, record_numbers):
        result = self._csv_data[column].loc[record_numbers]
        print("CsvHandler get_assignment_records result:", result)


# 欠損値補間 
   #行削除
    def drop_records(self, records_array):
        print("CsvHandler drop_records")
        self._csv_data.drop(records_array, axis=0, inplace=True)
    
    #行削除 範囲
    def drop_records_area(self, start, end):
        print("CsvHandler drop_records")
        self._csv_data.drop(self._csv_data.index[start:end], axis=0, inplace=True)

   #列削除
    def drop_columns(self, columns_array):
        print("CsvHandler drop_columns")
        self._csv_data.drop(columns_array, axis=1, inplace=True)

    #平均値取得（カラム指定）
    def average_value(self, column_name):
        print("CsvHandler average_value")
        result = self._csv_data[column_name].mean()
        print(result) 

   #平均値補間（カラム指定）
    def average_value_interpolation(self, column_name):
        print("CsvHandler average_value_interpolation")
        result = self._csv_data[column_name].fillna(self._csv_data[column_name].mean())
        print(result) 

    #中央値取得（カラム指定）
    def median_value(self, column_name):
        print("CsvHandler median_value")
        result = self._csv_data[column_name].median()
        print(result) 
    
    #中央値補間（カラム指定）
    def median_value_interpolation(self, column_name):
        print("CsvHandler median_value_interpolation")
        result = self._csv_data[column_name].fillna(self._csv_data[column_name].median())
        print(result) 
   
    #最頻値取得（カラム指定）
    def mode_value(self, column_name):
        print("CsvHandler mode_value")
        result = self._csv_data[column_name].mode()[0]
        print(result) 

    #最頻値補間（カラム指定）
    def mode_value_interpolation(self, column_name):
        print("CsvHandler mode_value_interpolation")
        result = self._csv_data[column_name].fillna(self._csv_data[column_name].mode()[0])
        print(result) 
            
   #小数点切り捨て（カラム指定）　←　平均と補完と組み合わせられないので要調査
    def decimal_point_truncation(self, column_name):
        print("CsvHandler decimal_point_truncation")
        result = self._csv_data[column_name].fillna(0).astype('int64')
        # result = np.floor(self._csv_data[column_name])
        print(result) 

# 相関係数
    # 1対1相関係数表示
    def correlation_coefficient_oto(self, column_name):
         corr_matrix = self._csv_data.select_dtypes(include='number').corr()
         result = corr_matrix[column_name]
         print(result) 

    # 1対多相関係数表示
    def correlation_coefficient_otm(self):
        result = self._csv_data.select_dtypes(include='number').corr()
        print(result) 
        # グラフ表示
        plt.figure(figsize=(8, 6))
        sns.heatmap(result, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.show()

# 多重代入法   
   #MICE
    def multiple_imputation_MICE(self, column_names):
        csv_data_selected = self._csv_data[column_names]

        mice_data = MICEData(csv_data_selected)
        for i in range(10):
            mice_data.update_all()
        result = mice_data.data
        print(result)

        
    #ランダムフォレスト
    def random_forest(self, column_array):
        print("ModelProcesrandom_forest column_array:", column_array)
        # age_df = self.data_instance.get_record(10)

        # 推定に使用する項目を指定
        age_df = self._csv_data[column_array]
        
        # ラベル特徴量をワンホットエンコーディング
        age_df=pd.get_dummies(age_df)

        # 学習データとテストデータに分離し、numpyに変換
        known_age = age_df[age_df.Age.notnull()].values  
        unknown_age = age_df[age_df.Age.isnull()].values

        # 学習データをX, yに分離
        X = known_age[:, 1:]  
        y = known_age[:, 0]

        # ランダムフォレストで推定モデルを構築
        rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
        rfr.fit(X, y)

        # 推定モデルを使って、テストデータのAgeを予測し、補完
        predictedAges = rfr.predict(unknown_age[:, 1::])
        self._csv_data.loc[(self._csv_data.Age.isnull()), 'Age'] = predictedAges 
        self.get_specification_record('Age', 890)
        
        # 年齢別生存曲線と死亡曲線
        facet = sns.FacetGrid(self._csv_data[0:890], hue="Survived",aspect=2)
        facet.map(sns.kdeplot,'Age',shade= True)
        facet.set(xlim=(0, self._csv_data.loc[0:890,'Age'].max()))
        facet.add_legend()
        plt.show()     




# 外れ値チェック


# 可視化
    # ヒストグラム
    def show_hist(self):
        self._csv_data.hist(figsize=(10, 10), bins=30)
        plt.show()

    # カーネル密度推定（KDE）
    def show_kds(self):
        sns.kdeplot(self._csv_data, shade=True)
        plt.show()
    
     # ペアプロット）
    def show_pair_plot(self):
        sns.pairplot(self._csv_data, diag_kind='kde')
        plt.show()
