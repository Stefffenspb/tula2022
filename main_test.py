# Библиотеки

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import preprocessing
import catboost
from catboost import CatBoostRegressor, Pool



# Подготовим данные
def data_prepare(df, df_sum):
    # ds = ds.astype(int)
    # ds =df['VISIT_MONTH_YEAR'] *100
    # ds = ds.astype(int)
    # ds = ds.astype(str).str[-2:]
    # ds = ds.astype(int)


    #df['VISIT_MONTH_YEAR'] = df['VISIT_MONTH_YEAR'].astype(int)
    # попробуем усечь до ковида
    # df = df.query("VISIT_MONTH_YEAR >= 2000")
    #df['VISIT_MONTH_YEAR'] = df['VISIT_MONTH_YEAR'].astype(str)
    #df = df.replace({'AGE_CATEGORY': {'young': '0.31', 'old': '0.83', 'middleage': '0.52', 'elderly': '0.67',
     #                                 'children': '0', 'centenarians': '1'}})
    df = pd.merge(df, df_sum, how='left', left_on=['PATIENT_SEX', 'MKB_CODE', 'ADRES','AGE_CATEGORY'], right_on=['PATIENT_SEX_F', 'MKB_CODE_F', 'ADRES_F','AGE_CATEGORY_F'])
    df['AGE_CATEGORY'] = df['proc']
    df = df.drop(columns='PATIENT_SEX_F')
    df = df.drop(columns='MKB_CODE_F')
    df = df.drop(columns='ADRES_F')
    df = df.drop(columns='AGE_CATEGORY_F')
    #df = df.drop(columns='VISIT_MONTH_YEAR_F')
    df = df.drop(columns='proc')
    df['AGE_CATEGORY'] = df['AGE_CATEGORY'].fillna('0')
    dm = df['VISIT_MONTH_YEAR'].astype(str).str[-2::]
    dt = df['VISIT_MONTH_YEAR'].astype(str).str[0:2]
    dm = dm.str.extract('(\w+)', expand=False)
    # dm = dm.astype(int)
    # print(dm)

    df['VISIT_MONTH_YEAR'] = dm + dt
    print(df.info)
    if 'PATIENT_ID_COUNT' in df:
        df['PATIENT_ID_COUNT'] = df['PATIENT_ID_COUNT'].abs()
        # df['PATIENT_ID_COUNT'] = df['PATIENT_ID_COUNT'].mask(df['PATIENT_ID_COUNT'] < 0, 0)

    df = df.fillna('0')

    # df = df.drop(columns='ADRES')

    # df = df.drop(columns='AGE_CATEGORY')
    # df = df.astype('float64')
    print(df.skew())
    print(df.describe().T)
    print(df.info())
    return df


# Обучение и Прогнозирование
def science(col, preds, filecsv_from, filecsv_to, colim, df_sum):
    df_train = df
    for diag in colim:
        df_train.drop(columns=diag)

    X = df_train.drop(col, axis=1)

    y = df_train[col]
    test_size = 0.2
    seed = 1

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=test_size, random_state=seed)

    pool_train = Pool(X_train, Y_train,
                      cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES'])
    pool_test = Pool(X_validation,
                     cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES'])

    models = []
    # Список Моделей
    ####################

    ''' grid = {'depth': [8],
                'learning_rate': [0.1],
                'l2_leaf_reg': [1],
                'iterations': [ 10000]

                  }
    model_CBR = CatBoostRegressor()
    grid_search_result = model_CBR.grid_search(grid,
                                           X=pool_train,
                                           plot=True)

    print(grid_search_result)
    input("Press Enter to continue...")
    '''
    import numpy as np

    ##########


    models.append(('MB', CatBoostRegressor(task_type='CPU', loss_function= 'RMSE', random_seed=1,l2_leaf_reg=0.2,learning_rate=0.1,depth=8, thread_count=-1, iterations=10000)))


    names = []

    best = []

    for name, model in models:

        modi = model.fit(pool_train)

        predi = modi.predict(pool_test)

        r2 = sklearn.metrics.r2_score(Y_validation, predi)

        names.append(name)

        print(r2)
        best.append(r2)
    index = best.index(max(best))

    # Подготовимся к предсказанию
    best_train = models[index][1]

    df_test = pd.read_csv('data/test_dataset_test.csv', sep=';', index_col=None,
                          dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                                 'AGE_CATEGORY': str})
    df_test.head()
    df_test_2 = pd.read_csv(filecsv_to, delimiter=';')
    df_test_2.head()

    df_merged = pd.concat([df_test, df_test_2], axis=1)
    df_test = data_prepare(df_test, df_sum)

    df_merged_id = pd.read_csv('data/test_dataset_test.csv', sep=';', index_col=None,
                               dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                                      'AGE_CATEGORY': str})
    df_merged_id.head()


    pool_train_solution = Pool(X, y,
                               cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES'])
    pool_test_solution = Pool(df_test,
                              cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES'])

    model_solution = CatBoostRegressor(task_type='CPU', loss_function= 'RMSE', random_seed=1,l2_leaf_reg=0.2,learning_rate=0.1,depth=8, thread_count=-1, iterations=10000)
    model_solution.fit(pool_train_solution)

    # Получение ответов

    y_pred_solution = model.predict(pool_test_solution)
    # Приведем к целому

    y_pred_solution.astype(int)

    # Формируем sample_solution для отправки на платформу
    test = df_merged_id
    test['PATIENT_ID_COUNT'] = y_pred_solution.astype(int)
    test['PATIENT_ID_COUNT'] = test['PATIENT_ID_COUNT'].abs()

    # Сохраняем в csv файл

    test.to_csv('result/sample_solution.csv', sep=';', index=None)
    return preds


df = pd.read_csv('data/train_dataset_train.csv', sep=';', index_col=None,
                 dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str, 'AGE_CATEGORY': str,
                        'PATIENT_ID_COUNT': int})
df.head()
print(df.info())
df = df.dropna(subset=['PATIENT_ID_COUNT'])
# Подготовим данные для соотношений между возрастными группами в каждом пункте

df_gr_age = df.groupby(by=['PATIENT_SEX','MKB_CODE','ADRES','AGE_CATEGORY']).sum().reset_index()
df_gr_full = df.groupby(by=['PATIENT_SEX','MKB_CODE','ADRES']).sum().reset_index()
df_gr_full.rename(columns={'PATIENT_SEX':'PATIENT_SEX_F'}, inplace=True)
df_gr_full.rename(columns={'MKB_CODE':'MKB_CODE_F'}, inplace=True)
df_gr_full.rename(columns={'ADRES':'ADRES_F'}, inplace=True)
#df_gr_full.rename(columns={'VISIT_MONTH_YEAR':'VISIT_MONTH_YEAR_F'}, inplace=True)
df_gr_full.rename(columns={'PATIENT_ID_COUNT':'PATIENT_ID_COUNT_F'}, inplace=True)
df_sum = pd.merge(df_gr_age, df_gr_full,  how='left', left_on=['PATIENT_SEX','MKB_CODE','ADRES'], right_on = ['PATIENT_SEX_F','MKB_CODE_F','ADRES_F'])
df_sum['proc']= df_sum['PATIENT_ID_COUNT']/df_sum['PATIENT_ID_COUNT_F']
df_sum = df_sum.drop(columns='PATIENT_SEX')
df_sum = df_sum.drop(columns='MKB_CODE')
df_sum = df_sum.drop(columns='ADRES')

df_sum = df_sum.drop(columns='PATIENT_ID_COUNT_F')
df_sum = df_sum.drop(columns='PATIENT_ID_COUNT')
df_sum.rename(columns={'AGE_CATEGORY':'AGE_CATEGORY_F'}, inplace=True)
df = data_prepare(df, df_sum)

# Соотношение Столбцов

# print(df.info())


# Построим Матрицу
# scatter_matrix(df)
# plt.show()
preds = []

filecsv_from = 'data/sample_solution.csv'
df_result = pd.read_csv(filecsv_from, delimiter=';')
df_result.head()
filecsv_to = 'result/sample_solution.csv'
df_result.to_csv(filecsv_to, index=False, sep=';')

column_list = pd.Series(['PATIENT_ID_COUNT'])
colim = column_list
colim.index = column_list
for col in column_list:
    colim = colim.drop(col)

    science(col, preds, filecsv_from, filecsv_to, colim, df_sum)

