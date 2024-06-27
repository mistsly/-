from factor_analyzer.factor_analyzer import calculate_kmo
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from full_net import train, prediction
import os
from Gene import main_gene


def main(opt='L'):
    """
    :param opt: 'L'代表成锭率，‘Rc’代表成材率，这个需要根据前端选择传参
    :return: Null
    """
    # todo
    model_path = f'{opt}_model_params.pth'
    if opt == 'L':
        df_ding = pd.read_csv('./ding.csv')
    elif opt == 'Rc':
        df_ding = pd.read_csv('./cai.csv')
    # 非法值处理
    median_value = np.median(df_ding[df_ding[opt] < 1][opt])  # 0.943
    df_ding.loc[df_ding[opt] == 1, opt] = median_value
    print(df_ding.shape)
    df_ding = df_ding.iloc[:, 1:]
    cor_mat = df_ding.corr()
    print(f"相关系数：{cor_mat.iloc[:, -1]}")
    # 可视化
    # pic_cor_mat = sns.heatmap(cor_mat)
    # plt.show()

    # KMO检验
    # 通常取值从0.6开始进行因子分析
    kmo_all, kmo_model = calculate_kmo(df_ding.iloc[:, :-1])
    print(f"KMO得分：{kmo_model}")

    # 标准化，但只有X
    X = df_ding.iloc[:, :-1].values
    Y = df_ding.iloc[:, -1].values

    scaler = StandardScaler()
    x_std = scaler.fit_transform(X)
    # 获取标准差和均值Gene用
    max_values = np.max(X, axis=0)
    min_values = np.min(X, axis=0)
    print("每列的最大值:", max_values)
    print("每列的最小值:", min_values)
    print('均值为：{}\n 标准差为：{}'.format(scaler.mean_, np.sqrt(scaler.var_)))
    max_df = pd.DataFrame([max_values])
    min_df = pd.DataFrame([min_values])
    mean_df = pd.DataFrame([scaler.mean_])
    var_df = pd.DataFrame([np.sqrt(scaler.var_)])
    combined_df = pd.concat([max_df, min_df, mean_df, var_df])
    combined_df.to_csv(f'{opt}_combined_values.csv', index=False)

    if os.path.exists(model_path):
        ret = prediction()
    #     todo
    else:
        train(x_std, Y, opt)


if __name__ == '__main__':
    main(opt='L')
    # main(opt='Rc')
