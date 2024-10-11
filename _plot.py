""" 机器学习作图相关的函数 """

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors, ticker
from ml_main import cal_yearly_seasonal

# 作图默认参数控制
rcParams["figure.dpi"] = 100  # 图片显示默认分辨率
rcParams['savefig.dpi'] = 300  # 图片保存默认分辨率
rcParams['savefig.transparent'] = True  # 图片保存透明背景
rcParams['font.size'] = 12  # 图片默认字号
rcParams['axes.unicode_minus'] = False  # 作图时正常显示符号
rcParams['axes.linewidth'] = 1.2  # spine 边框线宽
rcParams['font.sans-serif'] = 'Microsoft YaHei'  # 作图字体：微软雅黑（同时支持中英文）
for v in ['xtick', 'ytick']:
    rcParams[v + '.major.size'] = 6  # 主刻度线长
    rcParams[v + '.minor.size'] = 4  # 次刻度线长
    rcParams[v + '.major.width'] = 1.2  # 主刻度线宽
    rcParams[v + '.minor.width'] = 1.2  # 次刻度线宽


def plot_lc(dict_all_params: dict, dict_optimal_param: dict, path_png: str | bool, suptitle=''):
    """ 学习曲线作图

        dict_all_params: dict, 调参过程所有参数
        dict_optimal_param: dict, 每个超参数的最优值
        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-21 v1
    单进程
    """

    # 超参数列表
    list_hyper = list(dict_all_params.keys())

    # 画布设置
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6), dpi=100, sharex=False, sharey=True)
    ax = ax.flatten()

    n = 0
    for i in list_hyper:

        # all params
        series_i = dict_all_params[i]

        # plot
        series_i.plot.line(color='green', marker='o', ax=ax[n], zorder=1)

        # x轴Log
        if i in ['n_estimators', 'max_depth']:
            ax[n].set_xscale('log')

        # x_label、y_label
        ax[n].set_xlabel(i)

        # y_lim，y轴显示范围
        ax[n].set_ylim(0, 1)

        # 最优参数点
        optimal_x = dict_optimal_param[i]
        optimal_y = series_i[optimal_x]

        # 最优参数点标注
        ax[n].scatter(optimal_x, optimal_y, color='red', marker='o', zorder=2)

        # 垂线
        # ax[n].axvline(x=optimal_x, color='r', linewidth=1, ymin=0, ymax=optimal_y)

        # annotate注释位置
        # annotate_x = optimal_x
        # annotate_y = 0.1

        # annotate注释文本
        if i in ['max_features', 'max_samples']:
            annotate_text = '%.1f' % optimal_x
        else:
            annotate_text = '%d' % optimal_x

        # ax[n].arrow(x=optimal_x, y=0,
        #             dx=0, dy=optimal_y,
        #             width=0.1,
        #
        #             )

        # annotate注释
        # ax[n].annotate(
        #     # text=annotate_text,
        #     text='',
        #     xy=(optimal_x, 0),
        #     xytext=(annotate_x, optimal_y),
        #     # xytext=(annotate_x, annotate_y),
        #     ha='center',
        #     linespacing=5,
        #     color='red',
        #     arrowprops=dict(arrowstyle='-|>', facecolor='red', edgecolor='red'),
        # )

        # 最优参数注释
        ax[n].text(
            x=optimal_x,
            y=optimal_y + 0.05,
            s=annotate_text,
            color='red',
            ha='center',

        )

        n += 1

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    # 在最后一个图中显示score
    ax[5].text(x=0.55, y=0.1, s='Training Score：%.4f' % optimal_y, ha='center', color='black')

    # 显示yticklabels
    for n in range(6):
        ax[n].yaxis.set_tick_params(which='both', labelleft=True)

    # 显示ylabel
    for n in range(6):
        ax[n].set_ylabel('Score', visible=True, color='black')

    # plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.1, left=0.06, right=0.99, hspace=0.3, wspace=0.25)

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_performance(data_train: pd.Series, data_test: pd.Series, dict_train: dict, dict_test: dict, path_png: str | bool, suptitle=''):
    """ 模型性能作图

        data_train: 用于训练的数据
        data_test: 用于验证的数据
        dict_train: 模型预测的训练数据
            {
            'r2': float,
            'rmse': float,
            'data': pd.Series,
            }
        dict_test: 模型预测的验证数据，数据结构同上

        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-25 v1
    单进程
    """

    # 画布设置
    fig = plt.figure(figsize=(14, 8), dpi=100)

    ax0 = plt.subplot2grid((2, 3), loc=(0, 0), rowspan=1, colspan=2, fig=fig)
    ax1 = plt.subplot2grid((2, 3), loc=(0, 2), rowspan=1, colspan=1, fig=fig)
    ax2 = plt.subplot2grid((2, 3), loc=(1, 0), rowspan=1, colspan=2, fig=fig)
    ax3 = plt.subplot2grid((2, 3), loc=(1, 2), rowspan=1, colspan=1, fig=fig)

    ax = [ax0, ax1, ax2, ax3]

    # 时间序列图
    data_train.plot.line(color='grey', ax=ax[0], zorder=1, label='Observation', lw=1)
    dict_train['data'].plot.line(color='black', ax=ax[0], zorder=1, label='Prediction', lw=1)

    # 提取数据
    array1d_train = data_train.to_numpy()
    array1d_train_predicted = dict_train['data'].to_numpy()

    # 统计直方图
    ax_y1 = ax[1].twinx()
    ax_y1.hist(x=array1d_train, bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

    # 直方图y轴范围降到1/3
    ax_y1.set_ylim(0, ax_y1.get_ylim()[1] * 8)

    # 直方图yticks
    ax_y1.tick_params(labelright=False, right=False)

    # 相关性散点图
    ax[1].scatter(x=array1d_train, y=array1d_train_predicted, s=50, marker="$\u25EF$", alpha=0.8, color='red', lw=0.25)

    # 拟合结果slope, intercept, r, p, stderr_slope, stderr_intercept
    fitting_result_train = stats.linregress(x=array1d_train, y=array1d_train_predicted)

    # 斜率和截距
    slope_train, intercept_train = fitting_result_train[:2]

    # 1:1线端点
    y_11_train = (0, array1d_train.max())

    # 1:1线
    ax[1].plot((0, array1d_train.max()), y_11_train, color='black', lw=1.5, label='1:1')

    # 拟合线端点
    fitting_y_train = (slope_train * array1d_train.min() + intercept_train, slope_train * array1d_train.max() + intercept_train)

    # 拟合线
    ax[1].plot((array1d_train.min(), array1d_train.max()), fitting_y_train, color='green', lw=1.5, label='linear fitting')

    # 注释内容
    annotation_train = "${y=%.2fx + %.2f}$\n${R^2=%.2f}$\n${RMSE=%.2f}$" % (slope_train, intercept_train, dict_train['r2'], dict_train['rmse'])

    # 注释
    ax[1].text(x=0, y=ax[1].get_ylim()[1] * 0.95, s=annotation_train, color='green', ha='left', va='top')

    # xlabel、ylabel
    ax[0].set_ylabel(data_train.name)
    ax[1].set_xlabel('Observation')
    ax[2].set_xlabel('datetime')
    ax[1].set_ylabel('Prediction')
    ax[2].set_ylabel(data_train.name)
    ax[3].set_xlabel('Observation')
    ax[3].set_ylabel('Prediction')

    # 图例
    ax[0].legend(loc='best', frameon=False, framealpha=1, ncol=2)
    ax[1].legend(loc='lower right', frameon=False, framealpha=1)

    # 设置散点图在直方图上层
    ax[1].set_zorder(ax_y1.get_zorder() + 1)
    ax[1].patch.set_visible(False)

    if data_test is not None:
        # 时间序列图
        data_test.plot.line(color='grey', ax=ax[2], zorder=1, label='Observation', lw=1)
        dict_test['data'].plot.line(color='black', ax=ax[2], zorder=1, label='Prediction', lw=1)

        # 提取数据
        array1d_test = data_test.to_numpy()
        array1d_test_predicted = dict_test['data'].to_numpy()

        # 统计直方图
        ax_y3 = ax[3].twinx()
        ax_y3.hist(x=array1d_test, bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

        # 直方图y轴范围降到1/3
        ax_y3.set_ylim(0, ax_y3.get_ylim()[1] * 8)

        # 直方图yticks
        ax_y3.tick_params(labelright=False, right=False)

        # 相关性散点图
        ax[3].scatter(x=array1d_test, y=array1d_test_predicted, s=50, marker="$\u25EF$", alpha=0.8, color='red', lw=0.25)

        # 拟合结果slope, intercept, r, p, stderr_slope, stderr_intercept
        fitting_result_test = stats.linregress(x=array1d_test, y=array1d_test_predicted)

        # 斜率和截距
        slope_test, intercept_test = fitting_result_test[:2]

        # 1:1线端点
        y_11_test = (0, array1d_test.max())

        # 1:1线
        ax[3].plot((0, array1d_test.max()), y_11_test, color='black', lw=1.5, label='1:1')

        # 拟合线端点
        fitting_y_test = (slope_test * array1d_test.min() + intercept_test, slope_test * array1d_test.max() + intercept_test)

        # 拟合线
        ax[3].plot((array1d_test.min(), array1d_test.max()), fitting_y_test, color='green', lw=1.5, label='linear fitting')

        # 注释内容
        annotation_test = "${y=%.2fx + %.2f}$\n${R^2=%.2f}$\n${RMSE=%.2f}$" % (slope_test, intercept_test, dict_test['r2'], dict_test['rmse'])

        # 注释
        ax[3].text(x=0, y=ax[3].get_ylim()[1] * 0.95, s=annotation_test, color='green', ha='left', va='top')

        # 图例
        ax[2].legend(loc='best', frameon=False, framealpha=1, ncol=2)
        ax[3].legend(loc='lower right', frameon=False, framealpha=1)

        # 设置散点图在直方图上层
        ax[3].set_zorder(ax_y3.get_zorder() + 1)
        ax[3].patch.set_visible(False)

    # xticklabels旋转
    for k in range(4):

        # 对齐
        for tick in ax[k].xaxis.get_ticklabels():
            tick.set_horizontalalignment('center')

        # 旋转
        ax[k].tick_params(axis='x', which='major', labelrotation=0)

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)
    # ax[0].set_title(suptitle, x=0.5, y=1.01)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_performance_models(data: pd.DataFrame, path_png: str | bool):
    """ 模型性能作图

        data: index为模型名称，包含r2_train、rmse_train、r2_test、rmse_test列
        path_png: str, 图片保存路径

    无返回值
    2023-06-25 v1
    2023-09-01 v1.1 label精度: '%.2g(%.1f)' -> '%.3g(%.2f)'
    单进程
    """

    # 模型数量
    num_model = data.shape[0]

    height = math.ceil(0.26 * num_model + 3)

    # 画布设置
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, height))

    # 作图r2
    data.plot.barh(y=['r2_train', 'r2_test'], color=['silver', 'orange'], width=0.8, legend=False, ax=ax)

    # container
    container_train, container_test = ax.containers

    # 添加bar label —— r2（rmse）- train
    labels_train = ['%.3g(%.2f)' % (i[0], i[1]) for i in data.loc[:, ['r2_train', 'rmse_train']].to_numpy()]
    ax.bar_label(container=container_train, labels=labels_train, label_type='edge', fontsize=10)

    # 添加bar label —— r2（rmse）- test
    labels_test = []
    for i in data.loc[:, ['r2_test', 'rmse_test']].to_numpy():
        if not np.isnan(i[0]):
            labels_test.append('%.2g(%.1f)' % (i[0], i[1]))
        else:
            labels_test.append('')

    ax.bar_label(container=container_test, labels=labels_test, label_type='edge', fontsize=10, color='black')

    # xlabel
    ax.set_xlabel('${R^2}$')

    # xlim
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.1)

    # 子图标题
    ax.set_title('模型预测表现 - ${R^2(RMSE)}$', x=0.5, y=1)

    # 窗口标题
    fig.canvas.manager.set_window_title('performance')

    # 设置legend
    ax.legend(
        labels=['Train', 'Test'],
        loc='lower left',
        bbox_to_anchor=(0, 1),
        ncol=2,
        frameon=False,
        # fontsize=16,
        handlelength=1.5,  # 图例的长度
        handletextpad=0.5,  # 图例与文字的间距
        columnspacing=0.5,  # 图例列间距
        reverse=True,
    )

    plt.tight_layout()

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_dependence_model(data_shap: pd.DataFrame, data_train: pd.DataFrame, path_png: str | bool, suptitle=''):
    """ 浓度shapley values对应图

        data_shap: pd.DataFrame，Shapley values，含有datetime索引，所有列均为自变量
        data_train: pd.DataFrame, 训练数据（观测数据），含有和data_shap完全相同的datetime索引，最后一列为因变量，其他列为自变量
        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-25 v1
    单进程
    """

    # 自变量x
    list_x = data_train.columns[:-1]

    # 因变量
    y = data_train.columns[-1]

    # 作图行列数
    plot_rows = math.floor(len(list_x) ** 0.5)
    plot_cols = math.ceil(len(list_x) / plot_rows)

    # 画布设置
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(18, 10))

    if plot_rows * plot_cols == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    n = 0
    for x in list_x:
        # 散点map图
        scatter_n = ax[n].scatter(
            x=data_train.loc[:, x],  # x
            y=data_shap.loc[:, x],  # y
            s=60,  # 大小
            c=data_train[y],  # 颜色
            marker="$\u25EF$",  # 圆圈
            alpha=0.8,  # 透明度
            cmap='jet',  # 颜色映射
            lw=0.25,  # 线宽
            norm=colors.LogNorm(),  # cmap对数
        )

        # 统计直方图
        ax_in = ax[n].inset_axes(bounds=(0, 1.0, 1, 0.15), sharex=ax[n])
        ax_in.hist(x=data_train.loc[:, x], bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

        # 直方图关闭坐标轴，只保留数据
        ax_in.set_axis_off()

        # 轴标签
        ax[n].set_xlabel(x)
        ax[n].set_ylabel('shap')

        # colorbar
        # cb = fig.colorbar(scatter_n, ax=ax_yn, extend='neither')
        cb = fig.colorbar(scatter_n, ax=ax[n], extend='neither')
        # cb = fig.colorbar(scatter_n, ax=ax[n], extend='both')

        # colorbar标题
        cb.ax.set_title(y, fontsize=10)

        n += 1

    # 关闭多余的子图
    for i in range(len(list_x), plot_rows * plot_cols):
        ax[i].set_axis_off()

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)
    # ax[0].set_title(suptitle, x=0.5, y=1.01)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_dependence_x(data_shap: pd.DataFrame, data_train_x: pd.DataFrame, data_train_y: pd.DataFrame, path_png: str | bool, y: str, suptitle=''):
    """ 浓度shapley values对应图（按自变量分类）

        data_shap: pd.DataFrame，Shapley values，含有datetime索引，所有列均为自变量
        data_train_x: pd.DataFrame, 训练数据（观测数据），含有和data_shap完全相同的datetime索引，自变量
        data_train_y: pd.DataFrame, 训练数据（观测数据），含有和data_shap完全相同的datetime索引，因变量
        path_png: str, 图片保存路径
        y: 因变量名
        suptitle: str, 主标题

    无返回值
    2023-06-25 v1
    单进程
    """

    # 模型文件名
    list_name = data_shap.columns.to_list()

    # 作图行列数
    plot_rows = math.floor(len(list_name) ** 0.5)
    plot_cols = math.ceil(len(list_name) / plot_rows)

    # 画布设置
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(18, 10))
    if plot_rows * plot_cols == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    n = 0
    for m in list_name:
        # 散点map图
        scatter_n = ax[n].scatter(
            x=data_train_x.loc[:, m],  # x
            y=data_shap.loc[:, m],  # y
            s=60,  # 大小
            c=data_train_y.loc[:, m],  # 颜色
            marker="$\u25EF$",  # 空心圆圈
            alpha=0.8,  # 透明度
            cmap='jet',  # 颜色映射
            lw=0.25,  # 线宽
            norm=colors.LogNorm(),  # cmap对数
        )

        # 统计直方图
        ax_in = ax[n].inset_axes(bounds=(0, 1.0, 1, 0.15), sharex=ax[n])
        ax_in.hist(x=data_train_x.loc[:, m], bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

        # 直方图关闭坐标轴，只保留数据
        ax_in.set_axis_off()

        # 轴标签
        ax[n].set_xlabel(suptitle)
        ax[n].set_ylabel('shap')

        # colorbar
        cb = fig.colorbar(scatter_n, ax=ax[n], extend='neither')
        # cb = fig.colorbar(scatter_n, ax=ax[n], extend='both')

        # colorbar标题
        cb.ax.set_title(y, fontsize=10)

        # 子图标题
        ax[n].set_title(m)

        n += 1

    # 关闭多余的子图
    for i in range(len(list_name), plot_rows * plot_cols):
        ax[i].set_axis_off()

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)
    # ax[0].set_title(suptitle, x=0.5, y=1.01)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.95)

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_global(data_shap: pd.DataFrame, path_png: str | bool, suptitle=''):
    """ global shapley value作图

        data_shap: pd.DataFrame，索引为自变量，各列对应不同模型

        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-25 v1
    单进程
    """

    # 模型名列表
    list_filename = data_shap.columns.to_list()

    # 作图行列数
    plot_rows = math.floor(len(list_filename) ** 0.4)
    plot_cols = math.ceil(len(list_filename) / plot_rows)

    # 画布设置
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(20, 10), dpi=100, sharex=False, sharey=False)

    if not isinstance(ax, np.ndarray):
        ax = [ax]
    else:
        ax = ax.flatten()

    n = 0
    for name in list_filename:
        # 准备数据
        series_name = data_shap.loc[:, name].copy()

        # 排序
        # pd.Series().sort_values()
        series_name.sort_values(ascending=True, inplace=True)

        # 计算百分比
        series_name_percent = series_name / series_name.sum() * 100
        # print(series_name)

        # 作图
        ax_n = series_name.plot.barh(ax=ax[n], ylabel='', color='#f42756')

        # xlabel
        ax[n].set_xlabel('$\overline{\mathrm{|shap|}}$')
        # ax[n].set_xlabel('mean(|SHAP value|)')

        # 子图标题
        ax[n].set_title(name)

        # 设置label
        for container in ax_n.containers:
            labels = ['%.1f%%' % v for v in series_name_percent]
            ax[n].bar_label(container=container, labels=labels, fmt='%.1f', label_type='edge')

        # xlim
        ax[n].set_xlim((0, ax[n].get_xlim()[1] * 1.15))

        n += 1

    # 关闭多余子图
    for i in range(len(list_filename), plot_rows * plot_cols):
        ax[i].set_axis_off()

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_yearly_model(data_shap: pd.DataFrame, path_png: str | bool, suptitle=''):
    """ 作图：global shap年际变化

        data_shap: pd.DataFrame，Shapley values，含有datetime索引，所有列均为自变量
        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-26 v1
    单进程
    """

    # 数据按年分组
    group_yearly = data_shap.abs().groupby(by=data_shap.index.year)

    # 计算均值
    df_shap_yearly = group_yearly.mean()

    # 计算占比
    df_shap_yearly_percent = df_shap_yearly.T / df_shap_yearly.sum(axis=1)
    df_shap_yearly_percent = df_shap_yearly_percent.T * 100

    # 自变量
    list_x = df_shap_yearly.columns.tolist()

    # 作图行列数
    plot_rows = math.floor(len(list_x) ** 0.5)
    plot_cols = math.ceil(len(list_x) / plot_rows)

    # 作图
    ax = df_shap_yearly.plot.bar(
        width=0.8,
        # cmap='jet',
        color='black',
        subplots=True,
        figsize=(18, 10),
        layout=(plot_rows, plot_cols),
        sharex=False,
        sharey=False,
        title=suptitle,
        legend=False,
    )

    ax = ax.flatten()

    # 双y轴，占比图
    for i in ax:
        name_i = i.title.get_text()
        if name_i:
            ax_yi = i.twinx()

            # 作图
            ax_yi.scatter(
                x=i.get_xticks(),
                y=df_shap_yearly_percent.loc[:, name_i],
                s=80,
                marker='_',
                # marker='s',
                color='red',
                lw=2,
            )

            # ylabel
            ax_yi.set_ylabel('percent (%)', color='red')

            # 轴的颜色
            ax_yi.spines['right'].set_color('red')
            i.spines['right'].set_color('red')
            ax_yi.tick_params(axis='y', colors='red')

    # 关闭xlabel，设置ylabel
    for i in ax:
        i.set_ylabel('$\overline{\mathrm{|shap|}}$')
        i.set_xlabel('')

    # 窗口标题
    plt.gcf().canvas.manager.set_window_title(suptitle)

    plt.tight_layout()

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_yearly_x(data_shap: pd.DataFrame, data_shap_percent: pd.DataFrame, path_png: str | bool, suptitle=''):
    """ 作图：global shap年际变化，按自变量分类，不同模型放到一个图中

        data_shap: pd.DataFrame，Shapley values，含有datetime索引，所有列均为自变量
        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-25 v1
    单进程
    """

    # 列名
    list_col = data_shap.columns.tolist()

    # 作图行列数
    plot_rows = math.floor(len(list_col) ** 0.5)
    plot_cols = math.ceil(len(list_col) / plot_rows)

    # 作图
    ax = data_shap.plot.bar(
        width=0.8,
        # cmap='jet',
        color='black',
        subplots=True,
        figsize=(18, 10),
        layout=(plot_rows, plot_cols),
        sharex=False,
        sharey=False,
        title=suptitle,
        legend=False,
    )

    ax = ax.flatten()

    # 双y轴，占比图
    for i in ax:
        name_i = i.title.get_text()
        if name_i:
            ax_yi = i.twinx()

            # 作图
            ax_yi.scatter(
                x=i.get_xticks(),
                y=data_shap_percent.loc[:, name_i],
                s=80,
                marker='_',
                # marker='s',
                color='red',
                lw=2,
            )

            # ylabel
            ax_yi.set_ylabel('percent (%)', color='red')

            # 轴的颜色
            ax_yi.spines['right'].set_color('red')
            i.spines['right'].set_color('red')
            ax_yi.tick_params(axis='y', colors='red')

    # 关闭xlabel，设置ylabel
    for i in ax:
        i.set_ylabel('$\overline{\mathrm{|shap|}}$')
        i.set_xlabel('')

    # 窗口标题
    plt.gcf().canvas.manager.set_window_title(suptitle)

    plt.tight_layout()

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_yearly_seasonal(data_shap: dict, path_png: str | bool, suptitle=''):
    """ 作图：global shap分季节年际变化

        data_shap: dict，每个key对应一个子图，value为pd.DataFrame，Shapley values，含有year索引，所有列为四个季节
        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-26 v1
    单进程
    """

    # 子图标题
    list_title = list(data_shap.keys())

    # 包含的季节及颜色列表
    list_season = data_shap[list_title[0]].columns.tolist()
    dict_season_color = {
            'spring': 'green',
            'summer': 'red',
            'autumn': 'orange',
            'winter': 'blue',
    }
    list_color = [dict_season_color[i] for i in list_season]

    # 作图行列数
    plot_rows = math.floor(len(list_title) ** 0.5)
    plot_cols = math.ceil(len(list_title) / plot_rows)

    # 画布设置
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(18, 12), dpi=100, sharex=False, sharey=False)

    if plot_rows * plot_cols == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    # 作图
    n = 0
    for t in list_title:
        df_t = data_shap[t]

        df_t.plot.bar(
            ax=ax[n],
            width=0.8,
            # color=['green', 'red', 'orange', 'blue'],
            # cmap='jet',
            color=list_color,
            subplots=False,
            title=t,
            legend=False,
            stacked=False,
        )

        # ylabel
        ax[n].set_ylabel('$\overline{\mathrm{|shap|}}$')
        ax[n].set_xlabel('')

        n += 1

    # 设置公共legend
    fig.legend(
        # ax[0] + ax[1],
        labels=list_season,
        # ['spring', 'summer', 'autumn', 'winter'],
        loc='upper left',
        bbox_to_anchor=(0.05, 1.0),
        ncol=4,
        frameon=False,
        # fontsize=16,
        handlelength=1.5,  # 图例的长度
        handletextpad=0.5,  # 图例与文字的间距
        columnspacing=0.5,  # 图例列间距
    )

    # 关闭多余的子图
    for i in range(len(list_title), plot_rows * plot_cols):
        ax[i].set_axis_off()

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)
    # ax[0].set_title(suptitle, x=0.5, y=1.01)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    plt.tight_layout()

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_yearly_monthly(data_shap: dict, path_png: str | bool, suptitle=''):
    """ 作图：global shap分月年际变化

        data_shap: dict，每个key对应一个子图，value为pd.DataFrame，Shapley values，含有year索引，列对应月份
        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-26 v1
    单进程
    """

    # 子图标题
    list_title = list(data_shap.keys())

    # 月份及颜色列表
    list_monthly = data_shap[list_title[0]].columns.tolist()
    dict_month_color = {

            3: '#a1f192',
            4: '#43e426',
            5: '#217F10',

            6: '#fbb7b4',
            7: '#f86f69',
            8: '#F4271E',

            9: '#fac570',
            10: '#F7A829',
            11: '#e98a25',

            12: '#b7b7ff',
            1: '#8282ff',
            2: '#3000fb',
    }

    list_color = [dict_month_color[int(i)] for i in list_monthly]

    # 作图行列数
    plot_rows = math.floor(len(list_title) ** 0.5)
    plot_cols = math.ceil(len(list_title) / plot_rows)

    # 画布设置
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(20, 12), dpi=100, sharex=False, sharey=False)

    if plot_rows * plot_cols == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    # 作图
    n = 0
    for t in list_title:
        df_t = data_shap[t]

        df_t.plot.bar(
            ax=ax[n],
            width=0.8,
            # color=['green', 'red', 'orange', 'blue'],
            # cmap='jet',
            color=list_color,
            subplots=False,
            title=t,
            legend=False,
            stacked=False,
        )

        # ylabel
        ax[n].set_ylabel('$\overline{\mathrm{|shap|}}$')
        ax[n].set_xlabel('')

        n += 1

    # 设置公共legend
    fig.legend(
        labels=list_monthly,
        loc='upper left',
        bbox_to_anchor=(0.05, 1.0),
        ncol=len(list_monthly),
        frameon=False,
        # fontsize=16,
        handlelength=1.5,  # 图例的长度
        handletextpad=0.5,  # 图例与文字的间距
        columnspacing=0.5,  # 图例列间距
    )

    # 关闭多余的子图
    for i in range(len(list_title), plot_rows * plot_cols):
        ax[i].set_axis_off()

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)
    # ax[0].set_title(suptitle, x=0.5, y=1.01)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    plt.tight_layout()

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shap_global_percent_seasonal(data_shap: pd.DataFrame, path_png: str | bool, suptitle=''):
    """ global shapley value占比作图-分季节

        data_shap: pd.DataFrame，索引为自变量，各列对应不同模型

        path_png: str, 图片保存路径
        suptitle: str, 主标题

    无返回值
    2023-06-26 v1
    单进程
    """

    # 列名列表
    list_col = data_shap.columns.to_list()

    # 计算均值
    series_shap_mean_all = data_shap.mean(axis=0)

    # 计算占比
    series_shap_percent = series_shap_mean_all * 100 / series_shap_mean_all.sum()
    # series_shap_percent.sort_values(inplace=True, ascending=False)

    # 分季节统计
    series_shap_mean_spring = data_shap[data_shap.index.month.isin([3, 4, 5])].mean(axis=0)
    series_shap_mean_spring = series_shap_mean_spring * 100 / series_shap_mean_spring.sum()

    series_shap_mean_summer = data_shap[data_shap.index.month.isin([6, 7, 8])].mean(axis=0)
    series_shap_mean_summer = series_shap_mean_summer * 100 / series_shap_mean_summer.sum()

    series_shap_mean_autumn = data_shap[data_shap.index.month.isin([9, 10, 11])].mean(axis=0)
    series_shap_mean_autumn = series_shap_mean_autumn * 100 / series_shap_mean_autumn.sum()

    series_shap_mean_winter = data_shap[data_shap.index.month.isin([1, 2, 12])].mean(axis=0)
    series_shap_mean_winter = series_shap_mean_winter * 100 / series_shap_mean_winter.sum()

    # 合并
    df_total = pd.concat(
        objs=(
                series_shap_percent,
                series_shap_mean_winter,
                series_shap_mean_autumn,
                series_shap_mean_summer,
                series_shap_mean_spring,
        ),
        axis=1
    )
    df_total.columns = ['all', 'winter', 'autumn', 'summer', 'spring']

    # 作图
    ax = df_total.T.plot.barh(
        figsize=(18, 4),
        layout=(1, 1),
        width=0.8,
        cmap='tab20',
        subplots=False,
        # title=t,
        legend=False,
        stacked=True,
    )

    # 设置label
    for container in ax.containers:
        labels = ['%.1f' % v if v >= 1.3 else "" for v in container.datavalues]
        ax.bar_label(container=container, labels=labels, fmt='%.1f', label_type='center')

    # 设置公共legend
    plt.gcf().legend(
        # ax[0] + ax[1],
        # ['spring', 'summer', 'autumn', 'winter'],
        df_total.index,
        # loc='upper left',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.84),
        ncol=df_total.shape[0],
        frameon=False,
        # fontsize=16,
        # labelspacing=0,
        handlelength=1,  # 图例的长度
        handletextpad=0.5,  # 图例与文字的间距
        columnspacing=0.5,  # 图例列间距
    )

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)

    # x轴范围
    plt.xlim((0, 100))

    # xlabel
    plt.xlabel('$\overline{\mathrm{|shap|}}$ in percent (%)')

    # 窗口标题
    plt.gcf().canvas.manager.set_window_title(suptitle)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.95)

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_summary_x(data_train: pd.DataFrame, data_shap: pd.DataFrame, dir_png: str | bool, selected_x=None, suptitle=''):
    """ 汇总各个自变量

        data_train: pd.DataFrame, 训练数据（观测数据），含有和data_shap完全相同的datetime索引，最后一列为因变量，其他列为自变量
        data_shap: pd.DataFrame，Shapley values，含有datetime索引，所有列均为自变量
        dir_png: str, 图片保存路径
        selected_x: 只选择其中一个x作图
        suptitle: str, 主标题

    无返回值
    2023-06-26 v1
    单进程
    """

    print('作图 | summary_x | %s' % suptitle, end=' | ')

    # 计算mean(|shap|)、排序
    series_global_shap = data_shap.abs().mean(axis=0)
    series_global_shap.sort_values(ascending=True, inplace=True)

    # 计算percent、排序
    series_global_shap_percent = series_global_shap / series_global_shap.sum() * 100
    series_global_shap_percent.sort_values(ascending=True, inplace=True)

    # 数据按年分组平均，占比
    df_train_yearly = data_train.groupby(by=data_train.index.year).mean()
    df_shap_yearly = data_shap.abs().groupby(by=data_shap.index.year).mean()
    df_shap_yearly_percent = (df_shap_yearly.T / df_shap_yearly.sum(axis=1)).T * 100

    # 计算分季节数据
    dict_shap_seasonal = cal_yearly_seasonal(data=data_shap.abs())
    dict_train_seasonal = cal_yearly_seasonal(data=data_train)

    # 自变量列表
    if selected_x:
        list_x = [selected_x]
    else:
        list_x = data_shap.columns.tolist()

    # 因变量
    y = data_train.columns[-1]

    # 取年份数据
    list_common_year = [i for i in dict_train_seasonal[y].index if i in df_train_yearly.index]

    # 取月份数据
    list_month = list(map(str, sorted(list(set(data_train.index.month)), reverse=False)))

    for x in list_x:

        print(x, end=',')

        # x在series_global_shap中的位置
        loc_x = series_global_shap.index.get_loc(x)

        # 画布设置
        fig, axs = plt.subplot_mosaic(
            mosaic=[
                    ['a', 'b', 'c', 'd'],
                    ['a', 'e', 'f', '.'],
                    # ['a', 'd'],
            ],

            # layout='constrained',
            # layout='tight',
            height_ratios=[1, 1],
            width_ratios=[4, 4, 4, 4],
            figsize=(18, 8),
            # top=0.9,
        )

        # 窗口标题
        fig.canvas.manager.set_window_title("%s，y=%s，x=%s" % (suptitle, y, x))

        # 图像标题
        plt.suptitle("model=%s，y=%s，x=%s，year=%s~%s，month=%s" %
                     (suptitle, y, x, data_train.index[0].strftime('%Y'), data_train.index[-1].strftime('%Y'), ','.join(list_month)),
                     x=0.5, y=0.99,
                     )

        # global shap
        series_global_shap.plot.barh(ax=axs['a'], ylabel='', color='darkgrey')
        axs['a'].set_xlabel('$\overline{\mathrm{|shap|}}$')

        # 子图标题
        axs['a'].set_title('feature importance')

        # 设置bar label
        for container in axs['a'].containers:
            # labels = ['%.1f%%' % value for value in series_global_shap_percent]
            labels = ['%.2g(%.2g%%)' % (series_global_shap[i], series_global_shap_percent[i]) for i in range(series_global_shap.shape[0])]
            axs['a'].bar_label(container=container, labels=labels, fmt='%.1f', label_type='edge')

        # xlim
        axs['a'].set_xlim((0, axs['a'].get_xlim()[1] * 1.3))
        # print(df_global_shap_percent)

        # 突出x
        axs['a'].patches[loc_x].set_facecolor('#f42756')

        """ 年际变化 """
        # df_shap_yearly.loc[:, x].plot.bar(
        df_shap_yearly.loc[list_common_year, x].plot.bar(
            width=0.8,
            ax=axs['b'],
            color='dimgrey',
            # color='#b3e2cd',
            xlabel='',
        )

        # 双y轴-占比
        axs_by = axs['b'].twinx()
        axs_by.scatter(
            x=axs['b'].get_xticks(),
            # y=df_shap_yearly_percent.loc[:, x],
            y=df_shap_yearly_percent.loc[list_common_year, x],
            s=100,
            marker='_',
            # marker='s',
            color='red',
            lw=3,
        )

        # ylabel
        axs_by.set_ylabel('percent (%)', color='red')
        axs['b'].set_ylabel('$\overline{\mathrm{|shap|}}$')

        # 轴的颜色
        axs_by.spines['right'].set_color('red')
        axs['b'].spines['right'].set_color('red')
        axs_by.tick_params(axis='y', colors='red')

        """ global shap年际变化：分季节 """
        # dict_shap_seasonal[x].plot.bar(
        dict_shap_seasonal[x].loc[list_common_year, :].plot.bar(
            ax=axs['c'],
            width=0.8,
            color=['green', 'red', 'orange', 'blue'],
            # cmap='jet',
            # color='black',
            subplots=False,
            xlabel='',
            ylabel='$\overline{\mathrm{|shap|}}$',
            legend=False,
            stacked=False,
        )

        # b、c图y轴保持一致
        # axs['c'].set_ylim(axs['b'].get_ylim())

        # legend
        axs['c'].legend(
            ncol=2,
            frameon=False,
            # fontsize=10,
            handlelength=1,  # 图例的长度
            handletextpad=0.5,  # 图例与文字的间距
            columnspacing=0.5,  # 图例列间距
        )

        """ 自变量年际变化：分季节 """
        dict_train_seasonal[x].loc[list_common_year, :].plot.bar(
            ax=axs['e'],
            width=0.8,
            color=['green', 'red', 'orange', 'blue'],
            # cmap='jet',
            # color='black',
            subplots=False,
            xlabel='',
            ylabel=x,
            legend=False,
            stacked=False,
        )

        # 添加全年均值
        axs['e'].scatter(x=axs['e'].get_xticks(), y=df_train_yearly.loc[list_common_year, x], marker='*', color='black', s=100, label='均值')
        axs['e'].legend(
            labels=['均值'],
            ncol=1,
            frameon=False,
            # fontsize=10,
            handlelength=1,  # 图例的长度
            handletextpad=0.5,  # 图例与文字的间距
            columnspacing=0.5,  # 图例列间距
        )

        """ 因变量年际变化：分季节 """
        dict_train_seasonal[y].loc[list_common_year, :].plot.bar(
            ax=axs['f'],
            width=0.8,
            color=['green', 'red', 'orange', 'blue'],
            # cmap='jet',
            # color='black',
            subplots=False,
            xlabel='',
            ylabel=y,
            legend=False,
            stacked=False,
        )

        # 添加全年均值
        """ 因变量年际变化：分季节 """
        axs['f'].scatter(x=axs['f'].get_xticks(), y=df_train_yearly.loc[list_common_year, y], marker='*', color='black', s=100, label='均值')
        axs['f'].legend(
            labels=['均值'],
            ncol=1,
            frameon=False,
            # fontsize=10,
            handlelength=1,  # 图例的长度
            handletextpad=0.5,  # 图例与文字的间距
            columnspacing=0.5,  # 图例列间距
        )

        """ dependence """
        scatter_d = axs['d'].scatter(
            x=data_train.loc[:, x],  # x
            y=data_shap.loc[:, x],  # y
            s=60,  # 大小
            c=data_train.iloc[:, -1],  # 颜色
            marker="$\u25EF$",  # 圆圈
            alpha=0.8,  # 透明度
            cmap='jet',  # 颜色映射
            lw=0.25,  # 线宽
            norm=colors.LogNorm(),  # cmap对数
        )

        # 统计直方图
        ax_ind = axs['d'].inset_axes(bounds=(0, 1.0, 1, 0.1), sharex=axs['d'])
        ax_ind.hist(x=data_train.loc[:, x], bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

        # 直方图关闭坐标轴，只保留数据
        ax_ind.set_axis_off()

        # 轴标签
        axs['d'].set_xlabel(x)
        axs['d'].set_ylabel('shap')

        # colorbar
        ax_ind_cb = axs['d'].inset_axes(bounds=(1.02, 0, 0.04, 1))
        cb_d = fig.colorbar(scatter_d, cax=ax_ind_cb, extend='neither')

        # colorbar locator
        major_ticks_seed = np.array([1, 10, 100, 1000, 10000])
        minor_ticks_seed = np.arange(start=2, stop=10, step=1)
        minor_ticks = np.hstack([minor_ticks_seed * i for i in major_ticks_seed])

        major_ticks = [i for i in major_ticks_seed if data_train.iloc[:, -1].min() < i < data_train.iloc[:, -1].max()]
        minor_ticks = [i for i in minor_ticks if data_train.iloc[:, -1].min() < i < data_train.iloc[:, -1].max()]

        cb_d.ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
        cb_d.ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))

        # tick formatter
        minor_ticks_label = np.array([2, 3, 5, 7])
        minor_ticks_label = np.hstack([minor_ticks_label * i for i in major_ticks_seed])
        minor_ticks_label = [i if i in minor_ticks_label else '' for i in minor_ticks]

        cb_d.ax.yaxis.set_major_formatter(ticker.FixedFormatter(major_ticks))
        cb_d.ax.yaxis.set_minor_formatter(ticker.FixedFormatter(minor_ticks_label))

        # colorbar标题
        cb_d.ax.set_title(data_train.columns[-1], fontsize=10)

        # plt.tight_layout()
        # plt.subplots_adjust(top=0.93, bottom=0.1, right=0.95, hspace=0.23, wspace=0.5)
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96, hspace=0.23, wspace=0.45)

        if dir_png:
            path_png_x = os.path.join(dir_png, x + '.png')
            path_png_x = path_png_x.replace('*', '×')

            plt.savefig(path_png_x, transparent=True, dpi=300)
            plt.close()
        else:
            plt.show()

    print('完成！')
    # print('作图 | shap_dependence_x | %s' % 'df', flush=True)




# def plot_geo(name, values, longitude, latitude, path_html, value_range=[], splitnum=10, width='1200px', height='900px', title_legend='', precision=1, title='Title', subtitle='Subtitle'):
#     """
#     数据导入pyecharts作图, 地理信息数据作图
#     name: 站点名列表
#     values: 需要作图的数据列表
#     longitude, latitude:站点对应的经纬度
#     path_html: html保存路径
#     values_range:用于在legend中显示的范围
#     splitnum：如果不用连续颜色，则需指定legend分隔数量
#     title_legend: 图例标题
#     precision: 图例精度
#     title: 图形标题
#     subtitle: 图形副标题
#     """
#
#     from pyecharts.charts import Geo
#     from pyecharts.globals import GeoType
#     # from pyecharts.globals import ChartType
#     from pyecharts import options as opts
#     # from pyecharts.render import make_snapshot
#
#     """ 画布设置 """
#     # 初始化
#     geo = Geo(init_opts={'width': width, 'height': height})
#
#     # LabelOpts 控制省名的显示
#     geo.add_schema(maptype='china', zoom=1.2)
#
#     # 判断图例colorbar的显示范围
#     if value_range:
#         mi = value_range[0]
#         ma = value_range[1]
#     else:
#         mi = min(values)
#         ma = max(values)
#
#     # 自动分割pieces
#     list_pieces = []
#     array_pieces = np.linspace(mi, ma, splitnum + 1)
#     for i in range(array_pieces.shape[0] - 1):
#         dict_i = {'min': array_pieces[i], 'max': array_pieces[i + 1]}
#         list_pieces.append(dict_i)
#
#     # 图例设置
#     opts_legend = opts.VisualMapOpts(min_=mi, max_=ma,
#                                      orient='vertical',
#                                      #   pos_left='80%',
#                                      pos_bottom='23%',
#                                      pos_right='7%',
#                                      #   range_text=['(' + unit + ')', ''],
#                                      range_text=[title_legend, ''],
#                                      #   is_calculable=True,
#                                      range_opacity=0.8,
#                                      pieces=list_pieces,
#                                      is_piecewise=True,
#                                      precision=precision,  # 数据精度, 1为保留1位小数
#                                      range_color=['blue', 'cyan',
#                                                   '#00FF10', 'yellow', 'red'],
#                                      #   range_color=['#00FF10', 'yellow', 'red'],
#                                      textstyle_opts=opts.TextStyleOpts(
#                                          font_size=20),  # 图例字体设置
#                                      )
#
#     # 标题设置
#     opts_title = opts.TitleOpts(title=title,
#                                 subtitle=subtitle,
#                                 pos_left='2%',
#                                 pos_top='1%',
#                                 title_textstyle_opts=opts.TextStyleOpts(font_size=40),  # 标题格式
#                                 subtitle_textstyle_opts=opts.TextStyleOpts(font_size=28),  # 副标题格式
#                                 )
#
#     # 工具箱设置
#     opts_toolbox = opts.ToolboxOpts(orient='horizontal', pos_left='30%', pos_top='10%',
#                                     feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(type_='png',
#                                                                                                                      name=os.path.split(path_html)[1].replace('.html', ''),
#                                                                                                                      pixel_ratio=2.0,
#                                                                                                                      background_color='rgba(128, 128, 128, 0)'),
#                                                                     )
#                                     )
#
#     # 全局设置
#     geo.set_global_opts(visualmap_opts=opts_legend,
#                         title_opts=opts_title,
#                         toolbox_opts=opts_toolbox,
#                         )
#
#     """ 添加坐标数据 """
#     # 站点名称
#     for i in range(len(name)):
#         geo.add_coordinate(name[i], longitude[i], latitude[i])
#
#     values = [float(i) for i in values]
#     geo.add('', [(name[i], values[i]) for i in range(len(name))], type_=GeoType.SCATTER, symbol_size=10)  # 散点图
#
#     # 隐藏标记值
#     geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
#
#     # 执行作图
#     geo.render(path_html)
#
#     # 打开图片
#     # os.startfile(path=path_html)

if __name__ == '__main__':
    pass