import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import time

data_o = pd.read_csv('./data/industrial_data.csv')
test_o = pd.read_csv('./data/sampled_data.csv')
data_norm_o = pd.read_csv('./data/industrial_data.csv')
test_norm_o = pd.read_csv('./data/sampled_data.csv')

def function_3(p_m, x):
        a = np.dot(p_m, x)
        return a

def accuracy_rmse(selected_features):

    X = data_norm_o[selected_features].values
    y = data_norm_o['Target'].values

    # calculate parameter of linear model by QR-decomposition
    A = np.column_stack((np.ones(X.shape[0]), X))
    Q, R = np.linalg.qr(A)
    theta = np.linalg.solve(R, np.dot(Q.T, y))

    X = test_o[selected_features].values
    y = test_o['Target'].values
    A = np.column_stack((np.ones(X.shape[0]), X))
    #print('A: ', A)
    y_pred = np.dot(A, theta)
    # 計算 RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # 計算範圍 (Max - Min)
    range_y = np.max(y) - np.min(y)

    # 計算 Accuracy
    accuracy = 1 - rmse / range_y

    return accuracy


def accuracy_from_rmse(selected_features, theta):
    X = test_o[selected_features].values
    #print('selected: ', selected_features)
    #print('X: ', X)
    y = test_o['Target'].values

    A = np.column_stack((np.ones(X.shape[0]), X))
    #print('A: ', A)
    y_pred = np.dot(A, theta)
    # 計算 RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # 計算範圍 (Max - Min)
    range_y = np.max(y) - np.min(y)

    # 計算 Accuracy
    accuracy = 1 - rmse / range_y

    #print(f"RMSE: {rmse:.4f}")
    #print(f"Range: {range_y:.4f}")
    #print(f"Accuracy: {accuracy:.4f}")

    return accuracy

def calculate_MSE(selected_features, theta):

    X = test_o[selected_features].values
    y = test_o['Target'].values

    A = np.column_stack((np.ones(X.shape[0]), X))
    y_pred = np.dot(A, theta)
    mse_np = np.mean((y - y_pred) ** 2)

    return mse_np


def three_d_linear_regression(selected_features, norm=False, draw=False):
    """
    使用選擇的特徵進行 3D 線性回歸並計算 MSE。

    Args:
        selected_features: 選中的特徵名稱列表。
        norm: 是否使用標準化數據。
        draw: 是否繪製 3D 視覺化（限選 2 個特徵時）。

    Returns:
        mse_np: 計算的 MSE 值。
    """
    current_time = time.time()

    if norm:
        X = data_norm_o[selected_features].values
        y = data_norm_o['Target'].values
    else:
        X = data_o[selected_features].values
        y = data_o['Target'].values

    # calculate parameter of linear model by QR-decomposition
    A = np.column_stack((np.ones(X.shape[0]), X))
    Q, R = np.linalg.qr(A)
    theta = np.linalg.solve(R, np.dot(Q.T, y))

    # calculate MSE
    #y_pred = np.dot(A, theta)
    mse_np = calculate_MSE(selected_features, theta)
    #mse_np = np.mean((y - y_pred) ** 2)

    accuracy = accuracy_from_rmse(selected_features, theta)
    print(f"Accuracy based on RMSE: {accuracy}")

    # output ##################################################################
    #print(f"feature selected: {selected_features}")
    #print(f"model parameter with intercept(theta): {theta}")
    print(f"MSE: {mse_np}")
    #print(f"total time cost: {time.time() - current_time:.6f} 秒")

    # visualize when 2d dimension
    if draw and len(selected_features) == 2:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(data_o[selected_features[0]], data_o[selected_features[1]], data_o['Target'], label="True Data")
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_zlabel('Target')

        # 繪製回歸平面
        x_range = np.linspace(data_o[selected_features[0]].min(), data_o[selected_features[0]].max(), 50)
        y_range = np.linspace(data_o[selected_features[1]].min(), data_o[selected_features[1]].max(), 50)
        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
        Z_mesh = theta[0] + theta[1] * X_mesh + theta[2] * Y_mesh
        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color='r', alpha=0.5)
        plt.legend()
        plt.show()

    return mse_np
'''
selected_columns = ['Feature1', 'Feature2']
print("selected_columns : ", selected_columns)
three_d_linear_regression(selected_columns)
#three_d_linear_regression(selected_columns, True)
print()
'''

def fitness_function(features, norm=False):#input a particle contains all feature with 0 1
    if np.sum(features) == 0:  # 若無特徵被選中，返回較高的懲罰值
        return float('inf')  # 懲罰不選擇任何特徵的解

    # 根據粒子的選擇生成特徵名稱列表
    selected_columns = [f"Feature{i}" for i in range(1, 51) if features[i-1] == 1]
    
    # 如果選擇了特徵，則呼叫 three_d_linear_regression 計算 MSE
    if len(selected_columns) > 0:
        # 假設 three_d_linear_regression 會計算並返回 MSE
        #print('selected_column', selected_columns)
        mse = three_d_linear_regression(selected_columns)
        #print("test for now particle mse calculation: ", mse)
        return mse
    else:
        return float('inf')


# PSO实现
def pso_feature_selection(n, norm=False, max_iter=150, num_particles=50):
    np.random.seed(42)  # 设置随机种子
    num_features = 50  # 特征总数
    w_max, w_min = 0.9, 0.4  # 惯性权重范围
    c1, c2 = 1.5, 1.5  # 加速因子

    # 初始化粒子群
    #positions = np.random.randint(0, 2, (num_particles, num_features))  # set 0 or 1 on particle 2d array[[0, 1, ....],...[1, 0,....]] particles x features (30x50)
    positions = np.zeros((num_particles, num_features), dtype=int)  # 初始化为全 0
    for i in range(num_particles):
        # to avoid feature over n
        selected_indices = np.random.choice(num_features, n, replace=False)
        positions[i, selected_indices] = 1
    velocities = np.random.uniform(-1, 1, (num_particles, num_features))  # set velocties of particles
    personal_best_positions = positions.copy()  # local best position for each particles
    personal_best_scores = np.array([fitness_function(p) for p in positions])  # local best MSE, each p is like [0, 1, ....]
    #print("test personaL_best_scores: ", personal_best_scores)
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]  # find the minimun MSE value from each local particle to be global position
    global_best_score = np.min(personal_best_scores)  # global best MSE from global position
    
    #test_bestMSE = global_best_score
    #test_best = global_best_position
    
    
    #bestIteration = 0

    # PSO主循环
    for iteration in range(max_iter):
        w = w_max - (w_max - w_min) * (iteration / max_iter)  # dynamic weight

        
        
        for i in range(num_particles):
            # update particle velocity
            r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (personal_best_positions[i] - positions[i]) +
                c2 * r2 * (global_best_position - positions[i])
            )
            
            # 更新位置（用Sigmoid函数映射速度）#0 or 1
            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            positions[i] = np.where(np.random.rand(num_features) < sigmoid, 1, 0)
            
            # ensure feature count is n
            if np.sum(positions[i]) > n:
                indices = np.where(positions[i] == 1)[0]
                np.random.shuffle(indices)
                positions[i, indices[:np.sum(positions[i]) - n]] = 0
                #print("position now: ", positions[i])
            elif np.sum(positions[i]) < n:
                indices = np.where(positions[i] == 0)[0]
                np.random.shuffle(indices)
                positions[i, indices[:n - np.sum(positions[i])]] = 1
            
            # calculate MSE this iteration
            fitness = fitness_function(positions[i], norm)
            
            # update local best position
            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = positions[i].copy()
            
            # update global best position
            if fitness < global_best_score:#totally not called
                global_best_score = fitness
                global_best_position = positions[i].copy()
                #print("Iteration: ", iteration)
                #bestIteration = iteration + 1
            #print("bestIteration: ", bestIteration)
        #debug                                                         #check global MSE
        #print()
        #print(f"iteration: {iteration + 1}/{max_iter} - global best MSE：{global_best_score}")
        #print()

    # ensure global best solution has n features
    if np.sum(global_best_position) > n:
        indices = np.where(global_best_position == 1)[0]
        np.random.shuffle(indices)
        global_best_position[indices[:np.sum(global_best_position) - n]] = 0
    elif np.sum(global_best_position) < n:
        indices = np.where(global_best_position == 0)[0]
        np.random.shuffle(indices)
        global_best_position[indices[:n - np.sum(global_best_position)]] = 1

    #print("global position for the 0 iteration: ", test_best)
    #print("global best MSE for the 0 iteration: ", test_bestMSE)


    # 输出最终结果
    selected_features = [f"Feature{i}" for i in range(1, 51) if global_best_position[i-1] == 1]
    #print('test global: ', global_best_position)
    #print('test selected: ', selected_features)
    return selected_features, global_best_score

# 主程序

if __name__ == "__main__":

    '''
    n = 5  # 选取特征数
    pso_feature_selection(n, norm=False)
    selected_features, mse = pso_feature_selection(n, norm=False)
    print(f"final feature selected：{selected_features}")
    print(f"final best MSE：{mse}")

    print(f"Best features for n = {n}: {selected_features}")
    print(f"Best MSE for n = {n}: {mse}")
    accuracy = accuracy_rmse(selected_features)
    print(f"Best accuracy for n = {n}: {accuracy}")
    '''
    

    
    selected_columns = ['Feature1', 'Feature2', 'Feature15', 'Feature40', 'Feature44']
    print("selected_columns : ", selected_columns)
    three_d_linear_regression(selected_columns)
    print()
    #1 2 3 38 43
    #1 2 15 40 44
    

    '''
    selected_columns = ['Feature1']
    print("selected_columns : ", selected_columns)
    three_d_linear_regression(selected_columns)
    print()
    '''

    '''
    # 儲存每次 n 下的最佳結果
    best_features_per_n = {}
    best_mse_per_n = {}

    # 嘗試 1 到 50 個特徵
    for n in range(1, 51):
        start_time = time.time()  # 記錄開始時間

        print(f"Running PSO for n = {n}...")
        selected_features, mse = pso_feature_selection(n, norm=False)
        best_features_per_n[n] = selected_features
        best_mse_per_n[n] = mse

        end_time = time.time()  # 記錄結束時間
        elapsed_time = end_time - start_time  # 計算耗時
        accuracy = accuracy_rmse(selected_features)

        print(f"Best features for n = {n}: {selected_features}")
        print(f"Best MSE for n = {n}: {mse}")
        print(f"Best accuracy for n = {n}: {accuracy}")
        print(f"Time taken for n = {n}: {elapsed_time:.6f} seconds\n")
        print()

    # 找出全域最佳結果
    best_n = min(best_mse_per_n, key=best_mse_per_n.get)  # 找到 MSE 最小的 n
    final_selected_features = best_features_per_n[best_n]
    final_best_mse = best_mse_per_n[best_n]

    # 輸出結果
    print(f"Overall best number of features: {best_n}")
    print(f"Final feature selected for n = {best_n}: {final_selected_features}")
    print(f"Final best MSE: {final_best_mse}")
    '''
