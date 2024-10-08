import pandas as pd
import os
import numpy as np

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
        result = self._csv_data.info()
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
   
   #中央値
   #最頻値
   #中央値




# 外れ値チェック