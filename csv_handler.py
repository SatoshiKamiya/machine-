import pandas as pd
import os


# csvファイル処理クラス
class CsvHandler:
    # コンストラクタ
    def __init__(self, path):
        print("CsvHandler__init__ path:", path)
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

    # 各カラムの欠損値数チェック
    def get_all_missing_values_count(self):
        print("CsvHandler get_all_missing_values_count")
        result = self._csv_data.isna().sum()
        print(result)

     # 各カラムの欠損値数チェック
    def get_data_describe(self):
        print("CsvHandler get_data_describe")
        result = self._csv_data.describe()
        print(result)   

    # path取得
    def get_Path(self):
        return self.path


# 欠損値補間




# 外れ値チェック