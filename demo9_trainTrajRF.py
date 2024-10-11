""" Read data from csv file and train and test Random Forest Regression model """

from pathlib import Path
import winsound
import pandas as pd
import _rf as rf
import _hdf5 as hdf5
import _shap as shap
import _plot as plot


if __name__ == '__main__':

    # directory of project
    dir_project = Path(__file__).parent / 'ML'

    # path of csv files including raw data
    path_csv = dir_project / 'raw_data.csv'

    # filename without extension
    filename = path_csv.stem

    # path of saving hdf5 file
    path_h5 = dir_project / f'{filename}.h5'

    # path of saving model files
    path_model = dir_project / f'{filename}.skops'

    # path of saving learning curve to png files
    path_png_lc = dir_project / 'learning_curve.png'

    # path of saving performance to png files
    path_png_performance = dir_project / 'performance.png'

    # path of saving dependence figure to png files
    path_png_dependence = dir_project / 'dependence.png'
    
    # path of saving global shap values to png files
    path_png_global = dir_project / 'global.png'

    # datetime range for training and test
    dt_train = ('2015-01-01', '2020-12-31')
    dt_test = ('2021-01-01', '2022-12-31')

    # initial parameters of learning curve for training Random Forest Regression model
    dict_lc_params = {
            'n_estimators': [1, 2, 3, 4, 5, 10, 50, 100, 500],
            'max_depth': [1, 2, 3, 4, 5, 10, 20, 50, 100],
            'min_samples_split': [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
            'min_samples_leaf': [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # read csv file
    df_file = pd.read_csv(path_csv, index_col=0, parse_dates=True)

    # split data into training and test sets
    df_train = df_file.loc[dt_train[0]: dt_train[1], :]
    df_test = df_file.loc[dt_test[0]: dt_test[1], :]

    # saving training and test sets into hdf5 file
    hdf5.train2hdf(path_hdf5=path_h5, data=df_train)
    hdf5.test2hdf(path_hdf5=path_h5, data=df_test)

    # train and save result to hdf5 file
    rf.train_rfr_lc(
        init_params=dict_lc_params,
        data=df_train,
        path_skops=path_model,
        path_hdf5=path_h5,
        threshold=0.005,
        cv=10,
        cpu=-1,
        )
    
    # evaluate the performance - training set and test set
    dict_predict_training = rf.predict_rfr(path_skops=path_model, data=df_train)
    dict_predict_test = rf.predict_rfr(path_skops=path_model, data=df_test)

    # saving prediction results into hdf5 file
    hdf5.predict2hdf(path_hdf5=path_h5, dict_predict=dict_predict_training, location='Predict/Train')
    hdf5.predict2hdf(path_hdf5=path_h5, dict_predict=dict_predict_test, location='Predict/Test')

    # Calculate the SHAP value
    dict_shap = shap.cal_shap_rf(path_skops=path_model, data=df_train.iloc[:, :-1], cpu=-1)

    # save the SHAP value into hdf5 file
    hdf5.shap2hdf(dict_shap=dict_shap, path_hdf5=path_h5, location='Shapley/Train')

    # read hyperparameters from hdf5 file
    dict_all_params_i, dict_optimal_param_i = hdf5.hdf2lc(path_hdf5=path_h5, location='LearningCurve')

    # plot learning curve and save to png file
    plot.plot_lc(
        dict_all_params=dict_all_params_i, 
        dict_optimal_param=dict_optimal_param_i, 
        path_png=path_png_lc, 
        suptitle=filename
        )

    # plot performance and save to png file
    plot.plot_performance(
        data_train=df_train.iloc[:, -1],
        data_test=df_test.iloc[:, -1],
        dict_train=dict_predict_training,
        dict_test=dict_predict_test,
        path_png=path_png_performance,
        suptitle=filename,
        )
    
    # plot shapely dependence and save to png file
    plot.plot_shap_dependence_model(
        data_shap=dict_shap, 
        data_train=df_train, 
        path_png=path_png_dependence,
        suptitle=filename,
        )
    
    # plot global shap
    plot.plot_shap_global(data_shap=dict_shap, path_png=path_png_global, suptitle=filename)

    # beep
    winsound.Beep(300, 1000)
