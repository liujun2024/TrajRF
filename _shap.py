""" This package is for calculating SHAP values """

from __future__ import annotations
import os
import pandas as pd
from auto_shap.auto_shap import generate_shap_values
import skops.io as sio


def cal_shap_rf(path_skops: os.PathLike, data: pd.DataFrame, cpu=4):
    """ 计算SHAP values，随机森林模型

        path_skops: os.PathLike，模型的具体路径
        data: pd.DataFrame，用于计算SHAP values的自变量数据
        cpu: int，调用的CPU核心数量

        return: dict, {'shap_values_df': pd.DataFrame,
                       'shap_expected_value': float,
                       'global_shap_df': pd.Series,
                       }

    2023-06-19 v1
    单/多进程
    """

    # 检查data类型
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data数据类型错误，必须为pd.DataFrame！')

    # 读取模型
    model_rf = sio.load(file=path_skops)

    # print_time()
    # print('Calculating SHAP values...')

    # 调用函数计算
    shap_values_df, shap_expected_value, global_shap_df = generate_shap_values(model=model_rf, x_df=data, n_jobs=cpu, tree_model=True, regression_model=True)

    # 数据整理
    shap_values_df.index = data.index
    shap_values_df.index.name = 'datetime'

    global_shap_df = global_shap_df.set_index('feature', inplace=False).loc[:, 'shap_value']

    # print_time()
    # print('Done!')

    # 准备返回数据
    dict_result = {'shap_values_df': shap_values_df,
                   'shap_expected_value': shap_expected_value,
                   'global_shap_df': global_shap_df}

    return dict_result


if __name__ == '__main__':
    pass
