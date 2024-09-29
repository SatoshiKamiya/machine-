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
            self._csv_data = pd.read_csv(path)
        else:
            print(f"ファイルが見つかりません: {path}")

    # データチェック
    def checkHead(self):
        print("CsvHandler_checkHead")
        return self._csv_data.head()

    # カラム表示
    def checkColumn(self):
        print("CsvHandler_checkColumn")
        return self._csv_data.columns

    # path取得
    def getPath(self):
        return self.path
