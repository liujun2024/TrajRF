""" This package is for traning and testing Random Forest Regression model """

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from zipfile import ZIP_LZMA
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as optimizers
import skops.io as sio
from _hdf5 import lc2hdf


def create_rfr(dict_param: dict, cpu=1, **kwargs):
    """ 根据参数创建模型

        dict_param: 给定的参数，如：
                {'n_estimators': 1000,
                 'max_depth': 100,
                 'min_samples_split': 2,
                 'min_samples_leaf': 1,
                 'max_features': 'sqrt',
                 'max_samples': 1.0,
                 }

        x_train: pd.DataFrame，待训练的数据，自变量
        y_train: pd.Series，待训练的数据，因变量
        cpu: int，调用的CPU核心数量，默认为：1

        return: model

    """

    # 查看kwargs中的变量
    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose = 0

    # 建立随机森林回归模型
    model_rfr = RandomForestRegressor(
        n_estimators=dict_param['n_estimators'],  # 决策树数量
        criterion="squared_error",  # criterion：'squared_error' = mse, 'absolute_error' = mae, 'poisson'
        max_depth=dict_param['max_depth'],  # None：默认最大深度，最高复杂度
        min_samples_split=dict_param['min_samples_split'],  # 节点样本数量小于2时不再分支，最高复杂度
        min_samples_leaf=dict_param['min_samples_leaf'],  # 叶子节点所包含的最小样本数，最高复杂度
        min_weight_fraction_leaf=0.0,  # 0：不同样本间的权重一致
        max_features=dict_param['max_features'],  # max_features：'sqrt', 'log2', None}, int or float, default=1.0即max_features=n_features，最高复杂度
        max_leaf_nodes=None,  # 不限制叶子节点的个数
        min_impurity_decrease=0.0,  #
        bootstrap=True,  # https://stackoverflow.com/questions/40131893/random-forest-with-bootstrap-false-in-scikit-learn-python
        # If bootstrap is True, the number of samples to draw from X to train each base estimator， 默认最高复杂度
        oob_score=False,  #
        n_jobs=cpu,  #
        random_state=42,  #
        verbose=verbose,  # 控制台输出信息丰富程度
        warm_start=False,  #
        ccp_alpha=0.0,  #
        max_samples=dict_param['max_samples'],

    )

    return model_rfr


def score_rfr(dict_param: dict, x_train: pd.DataFrame, y_train: pd.Series, cv=10, cpu=0, **kwargs):
    """ 单点调参，给定一组参数，训练一个模型，输出模型表现

        dict_param: 给定的参数，如：
                {'n_estimators': 1000,
                 'max_depth': 100,
                 'min_samples_split': 2,
                 'min_samples_leaf': 1,
                 'max_features': 'sqrt',
                 'max_samples': 1.0,
                 }

        x_train: pd.DataFrame，待训练的数据，自变量
        y_train: pd.Series，待训练的数据，因变量
        cv: int，交叉验证，cv-fold
        cpu: int，调用的CPU核心数量，默认为：0，根据n_estimator自动调整cpu
        verbose: int，控制台输出信息丰富程度

        return: score   # 模型表现score

    2023-06-19 v1
    单/多进程
    """

    # 判断输入的参数是否齐全
    list_p = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_samples']
    if not all([i in dict_param.keys() for i in list_p]):
        raise ValueError('dict_param包含的参数不全！')

    if cpu == 0:
        cpu = min((math.ceil(dict_param['n_estimators'] / 5), 8))

    # 建立随机森林回归模型
    model_rfr = create_rfr(dict_param=dict_param, cpu=cpu, **kwargs)

    # 模型表现评估，https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    score = optimizers.cross_val_score(
        estimator=model_rfr,
        X=x_train,
        y=y_train,
        cv=cv,  # 交叉验证 cv-fold
        n_jobs=cpu,
    ).mean()

    # 返回数据
    return score


def tune_rfr_manual(dict_params: dict, x_train: pd.DataFrame, y_train: pd.Series, cv=10, cpu=1, **kwargs):
    """ 给定一组参数，评估其中含有多个值的参数(min_samples_split)的模型表现，将模型表现全部输出
        多个值的参数仅限：'min_samples_split', 'min_samples_leaf', 'max_features', 'max_samples'中的一个

        dict_params: 给定的参数，如
                {'n_estimators': 1000,
                 'max_depth': 100,
                 'min_samples_split': [2, 4, 6, 8, 10],
                 'min_samples_leaf': 1,
                 'max_features': 'sqrt',
                 'max_samples': 1.0,
                 }

        x_train: pd.DataFrame，待训练的数据，自变量
        y_train: pd.Series，待训练的数据，因变量
        cv: int，交叉验证，cv-fold
        cpu: int，调用的CPU核心数量，默认为：1

        return: dict，{
                'min_samples_split': int,  # 最优的min_samples_split值
                'score': float, #  对应的模型性能
                'data': pd.Series, 训练数据，index为评估对象的（如：min_samples_split）的输入值，data为模型表现score，name为评估对象名（min_samples_split）
                }

    2023-06-19 v1
    单/多进程
    """

    # 检查哪个参数是评估对象
    for i in dict_params.keys():
        if isinstance(dict_params[i], list) or isinstance(dict_params[i], np.ndarray):
            target_name = i
            break

    # 其它参数名列表
    list_other_name = [i for i in dict_params.keys() if i != target_name]

    # 创建模型，训练，并评估
    list_score = []
    for i in dict_params[target_name]:
        print('%.3g' % i, end='，')

        # 准备参数
        dict_i = {key: dict_params[key] for key in list_other_name}
        dict_i.update({target_name: i})

        # 调用函数评估模型表现
        score_i = score_rfr(dict_param=dict_i, x_train=x_train, y_train=y_train, cv=cv, cpu=cpu, **kwargs)

        # 结果存入列表
        list_score.append(score_i)

    # 创建返回数据，pd.Series
    series_result = pd.Series(data=list_score, index=dict_params[target_name], name=target_name)

    # 准备返回数据
    dict_result = {
            target_name: series_result.index[series_result.argmax()],
            'score': series_result.iloc[series_result.argmax()],
            'data': series_result,
    }

    # 返回数据
    return dict_result


def tune_rfr_automatic(init_param: dict, x_train: pd.DataFrame, y_train: pd.Series, threshold=0.005, cv=10, cpu=1, **kwargs):
    """ 给定一组参数，自动迭代评估其中含有多个初始值的参数(n_estimators)的模型表现，将模型表现全部输出
            多个值的参数仅限：'n_estimators', 'max_depth'中的一个，下面以n_estimators为例进行说明，

            init_param: 给定的初始参数，如
                    {'n_estimators': [1, 5, 10, 50, 100, 500],
                     'max_depth': 100,
                     'min_samples_split': 2,
                     'min_samples_leaf': 1,
                     'max_features': 'sqrt',
                     'max_samples': 1.0,
                     }

            threshold: float，阈值，默认为：0.005，用于判断n_estimators对应的模型表现衰减量，如果超过阈值，则需要继续添加n_estimators值进行迭代

            x_train: pd.DataFrame，待训练的数据，自变量
            y_train: pd.Series，待训练的数据，因变量
            cv: int，交叉验证，cv-fold
            cpu: int，调用的CPU核心数量，默认为：1
            verbose: int，控制台输出信息丰富程度

            return: dict，{
                n_estimators: int,  # 最优的n_estimators值
                'score': float, #  对应的模型性能
                'data': pd.Series, 训练数据，索引为评估对象的（如：n_estimators）的输入值及迭代值，值为模型表现score，name为评估对象名（n_estimators）
            }

        2023-06-19 v1
        单/多进程
        """

    # 检查哪个参数是评估对象
    for i in init_param.keys():
        if isinstance(init_param[i], list) or isinstance(init_param[i], np.ndarray):
            target_name = i
            break

    # 其它参数名列表
    list_other_name = [i for i in init_param.keys() if i != target_name]

    # n_estimators的初始值，如：[1, 5, 10, 50, 100, 500]
    list_x = list(init_param[target_name])

    """ 迭代评估，先评估初始值的模型表现，理论上模型表现与n_estimators和max_depth正相关，
        然后根据曲线（model score vs. n_estimators）的增长情况，选择增加n_estimators的迭代值，
        以符合迭代条件 """

    # 在给定n_estimators值以外的迭代次数，默认不大于4次
    count = 0

    # 用于保存训练过的n_estimators值
    list_x_trained = []

    # 用于保存(n_estimators值, score)
    list_score = []
    while len(list_x) > 0:

        # 从list_x中选择第一个值（如：[1, 5, 10, 50, 100, 500]中的1）进行评估
        n = list_x[0]

        print(n, end='，')

        # 准备参数
        dict_n = {key: init_param[key] for key in list_other_name}
        dict_n.update({target_name: n})

        # 评估模型表现
        score_n = score_rfr(dict_param=dict_n, x_train=x_train, y_train=y_train, cv=cv, cpu=cpu, **kwargs)

        # 结果存入列表
        list_score.append((n, score_n))

        # 从list_x中移除已经评估的值
        list_x.remove(n)
        list_x_trained.append(n)

        # 待将初始值评估完后判断是否需要向list_x中继续添加数据
        if len(list_x) != 0:
            continue
        else:

            # 现有结果生成pd.Series，index为n_estimators的值，data为模型表现score，name为target_name
            series_result = pd.Series(data=[i[1] for i in list_score], index=[i[0] for i in list_score], name=target_name)

            # 按照索引值（target_name对应的值）从小到大排序
            series_result.sort_index(ascending=True, inplace=True)

            # 模型性能最好时对应的位置索引
            index_max_score = series_result.argmax()

            # 模型性能最好时的score
            max_score = series_result.iloc[index_max_score]

            # 从n_estimators最小值循环至最大值
            for i in range(series_result.shape[0] - 1):
                # 第i个n_estimators值相比n_estimator最大值的衰减比
                change_percent_low = (max_score - series_result.iloc[i]) / max_score

                # 第i+1个n_estimators值相比n_estimator最大值的衰减比
                change_percent_high = (max_score - series_result.iloc[i + 1]) / max_score

                # 判断两个衰减比的位置，如果刚好在阈值两边，则需要在此范围内继续添加n_estimator值，默认为相同间隔的3个值
                if change_percent_low > threshold >= change_percent_high:
                    # 生成需要添加进list_x的值
                    list_add_x = np.linspace(series_result.index[i], series_result.index[i + 1], 5)[1: -1]

                    # 四舍五入
                    list_add_x = [round(i) for i in list_add_x]

                    # 去重
                    list_add_x = sorted(list(set(list_add_x)), reverse=False)

                    # 去除已经训练过的
                    list_add_x = [i for i in list_add_x if i not in list_x_trained]

                    if list_add_x:
                        # 添加进list_x
                        list_x += list_add_x

                    break

            """如果达到预期的迭代次数，或者list_x中所有值均已评估，则返回数据"""
            if count == 4 or len(list_x) == 0:
                # 最优的n_estimators
                best_x = series_result.index[i + 1]

                # 对应的模型性能
                best_score = series_result[best_x]

                # 准备返回数据
                dict_result = {
                        target_name: best_x,
                        'score': best_score,
                        'data': series_result,
                }

                return dict_result

            count += 1


def train_rfr_lc(init_params: dict | None, data: pd.DataFrame, path_skops: str, path_hdf5: str, threshold=0.005, cv=10, cpu=0, **kwargs):
    """ 模型训练：随机森林回归，训练方式：学习曲线

        init_params: 学习曲线初始参数，默认值：
                {'n_estimators': [1, 5, 10, 50, 100, 500, 1000],
                 'max_depth': [1, 2, 5, 10, 20, 50, 100],
                 'min_samples_split': list(range(2, 20, 2)),
                 'min_samples_leaf': list(range(1, 20, 2)),
                 'max_features': np.linspace(0.1, 1, 10),
                 'max_samples': np.linspace(0.1, 1, 10),
                 }

        data: pd.DataFrame，待训练数据，含有索引，最后一列为因变量，其它列为自变量
        path_skops: os.PathLike，模型保存路径
        path_hdf5: os.PathLike，模型训练结果保存路径
        threshold: 详见tune_rfr_automatic函数
        cv: int，交叉验证，k-fold
        cpu: int，训练调用的CPU核心数

    无返回值
    2023-06-19 v1
    单/多进程
    """

    # 学习曲线初始参数
    if init_params is None:
        init_params = {
                'n_estimators': [1, 5, 10, 50, 100, 500],
                'max_depth': [1, 2, 5, 10, 20, 50, 100],
                'min_samples_split': np.arange(start=2, stop=21, step=2),
                'min_samples_leaf': np.arange(start=1, stop=20, step=2),
                'max_features': np.linspace(0.1, 1, 10),
                'max_samples': np.linspace(0.1, 1, 10),
        }

    # print('初始参数:')
    # pprint(init_params)

    # 数据类型检查
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data数据类型错误，必须为pd.DataFrame！')

    # 自变量
    df_x_obs = data.iloc[:, :-1]
    # x = df_x_train.columns.tolist()

    # 因变量
    series_y_obs = data.iloc[:, -1]
    # y = series_y_train.name

    """ 开始训练，依次对n_estimators、max_depth、
        min_samples_split、min_samples_leaf、max_features、max_samples进行调参 """

    list_p = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_samples']

    # 判断list_p中的元素是否都存在于输入的init_params中
    if not all([i in init_params.keys() for i in list_p]):
        raise ValueError("init_args未包含以下所有键：\n%s" % ', '.join(list_p))

    # 用于储存最优参数，key为list_p中的元素，value为对应的最优值
    dict_best_param = dict()

    # 用于储存训练过程的所有结果，key为list_p中的元素，value为对应的pd.Series（index为参数值，data为对应的模型表现）
    dict_all_params = dict()

    # 过拟合初始参数-通常可以得到很好的模型表现
    params_overfitting = {
            'n_estimators': 1000,
            'max_depth': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'max_samples': 1.0,
    }

    ''' 对list_p中的参数依次训练 '''
    for p in list_p:

        print('训练：%s | ' % p, end='')

        # 准备参数
        init_param_p = dict_best_param
        init_param_p.update({p: init_params[p]})
        init_param_p.update({key: params_overfitting[key] for key in params_overfitting.keys() if key not in init_param_p.keys()})

        print(init_param_p, end='\n\t')

        # 调用函数进行训练
        if p in ['n_estimators', 'max_depth']:
            result_p = tune_rfr_automatic(init_param=init_param_p,
                                          x_train=df_x_obs,
                                          y_train=series_y_obs,
                                          threshold=threshold,
                                          cv=cv,
                                          cpu=cpu,
                                          **kwargs,
                                          )

        else:
            result_p = tune_rfr_manual(dict_params=init_param_p,
                                       x_train=df_x_obs,
                                       y_train=series_y_obs,
                                       cv=cv,
                                       cpu=cpu,
                                       **kwargs,
                                       )

        # 参数储存至字典
        dict_best_param[p] = result_p[p]
        dict_all_params[p] = result_p['data']

        print('完成！')

    """ 保存模型 """
    print('模型保存...', end=' ')

    # 生成模型
    model_rfc = create_rfr(dict_param=dict_best_param, cpu=4, **kwargs)
    model_rfc.fit(X=df_x_obs, y=series_y_obs)

    # 保存
    sio.dump(obj=model_rfc, file=path_skops, compression=ZIP_LZMA, compresslevel=3)
    print('完成！')

    """ 保存调参数据至hdf5文件 """
    print('学习曲线保存...', end=' ')
    lc2hdf(path_hdf5=path_hdf5, dict_all_params=dict_all_params, dict_best_param=dict_best_param, location='LearningCurve/')
    print('完成！')


def predict_rfr(path_skops: str, data: pd.DataFrame):
    """ 使用随机森林模型进行预测

        path_skops: os.PathLike，模型文件的具体路径
        data: pd.DataFrame，待预测的数据，含有索引，最后一列为因变量，其它列为自变量

        return：dict，
            {'r2': r2,
             'rmse': rmse,
             'data': series_predict,
             }

    2023-06-19 v1
    单进程
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError('data数据类型错误，必须为pd.DataFrame！')

    # 读取模型
    model_rfr = sio.load(file=path_skops)

    # 分离自变量和因变量
    x_obs = data.iloc[:, :-1]
    y_obs = data.iloc[:, -1]

    # 预测
    y_predict = model_rfr.predict(X=x_obs)

    # 计算root mean squared error均方根误差
    rmse = metrics.mean_squared_error(y_true=y_obs, y_pred=y_predict, squared=False)
    # mse = metrics.mean_squared_error(y_true=y_train, y_pred=y_predict, squared=True)

    # computes the coefficient of determination, usually denoted as R2
    r2 = metrics.r2_score(y_true=y_obs, y_pred=y_predict)

    # 准备返回数据
    series_predict = pd.Series(data=y_predict, index=data.index, name='predict')

    dict_result = {
            'r2': r2,
            'rmse': rmse,
            'data': series_predict,
    }

    # 返回数据
    return dict_result


if __name__ == "__main__":
    pass
