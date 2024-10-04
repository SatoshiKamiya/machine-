import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from csv_handler import CsvHandler


def main():
    print("main処理開始")
    csvHandler = CsvHandler("data/train.csv")
    # csvHandler.get_record(20)
    # csvHandler.drop_records([2,6,8])
    # csvHandler.drop_records_area(2,8)
    count = csvHandler.get_records_count()
    # csvHandler.get_specification_record(['Age'], count)
    # csvHandler.average_value_interpolation('Age')
    csvHandler.decimal_point_truncation('Age') # ←average_value_interpolationと組み合わせるとうまく起動しない


if __name__ == "__main__":
    main()
