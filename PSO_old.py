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

'''def three_d_linear_regression(selected_columns, norm = False, draw = True):
    current_time = time.time()
    if norm:
        scaler = StandardScaler()
        # data = scaler.fit_transform(data_o[need[:-1]]) #for first time to stardardize data
        data = data_norm_o[selected_columns[:-1]]
        draw_d = np.column_stack((data, data_o['Target']))
        draw_d = pd.DataFrame(draw_d, columns=selected_columns)
        test = test_norm_o[selected_columns[:-1]]
        test = test.values

        A = np.column_stack((np.ones(data.shape[0]), data))
        b = data_o['Target']
    else:#feature selected to do linear selection
        draw_d = data_o[selected_columns]
        data = data_o[selected_columns]
        data = data.values
        test = test_o[selected_columns[:-1]]#a few data from original dataset
        test = test.values

        #separate feature and target to A(feature)x=b(target)
        A = data[:, :-1]
        #print('A:',A)
        A = np.column_stack((np.ones(A.shape[0]), A))#push a 1 column to A
        b = data[:, -1]
        #print('B:',b)

    if draw:#basically just for 3-dimesion(2 features with target)
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        # ax1.scatter(draw_d[need[0]], draw_d[need[1]], draw_d[need[2]])
        ax1.set_xlabel(selected_columns[0])
        ax1.set_ylabel(selected_columns[1])
        ax1.set_zlabel(selected_columns[2])


    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))


    if not norm and draw:#it will auto scaling
        print("not norm")
        p_m = np.array([[0, 0]])
        x_p,y_p,_ = data_o[selected_columns].max().astype(int)
        for i in np.arange(0, x_p+1, 0.5):
            for j in np.arange(0, y_p+1, 0.5):
                p_m = np.vstack([p_m, [i, j]])
    elif draw:
        print("norm")
        p_m = np.array([[0, 0]])
        x_p,y_p,_ = draw_d[selected_columns].max().astype(int)
        for i in np.arange(-2, x_p+1, 0.05):
            for j in np.arange(-2, y_p+1, 0.05):
                p_m = np.vstack([p_m, [i, j]])

    print("x : ", x) #parameter for the linear model
    test = np.column_stack((np.ones(test.shape[0]), test))
    ans = np.dot(test, x)
    # print("ans : ", ans)

    if draw:#draw for test data
        p_z = function_3(p_m, x[1:]) + x[0]
        ax1.plot(p_m[:,0], p_m[:,1], p_z, c='r', alpha=0.5)
        ax1.scatter(test[:, 1], test[:, 2], ans, c='r')
        ax1.text(test[:, 1][0], test[:, 2][0], ans[0], f'{ans[0]:.1f}', None)
        ax1.scatter(test[:, 1], test[:, 2], test_o[selected_columns[-1]], c='purple')
        ax1.text(test[:, 1][0], test[:, 2][0], test_o[selected_columns[-1]][0], f'{test_o[selected_columns[-1]][0]:.1f}', None)
    
    #data = np.column_stack((np.ones(data.shape[0]), data))
    #ans = np.dot(data, x)
    mse_np = np.mean((data_o[selected_columns[-1]] - ans)**2)#calculate mes for all data
    print("MSE using numpy's solution: ", mse_np)
    print("cost time : ", time.time() - current_time)
'''
def three_d_linear_regression(selected_features, norm=False, draw=True):
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
    y_pred = np.dot(A, theta)
    mse_np = np.mean((y - y_pred) ** 2)

    # 輸出結果
    print(f"feature selected: {selected_features}")
    print(f"model parameter with intercept(theta): {theta}")
    print(f"global MSE: {mse_np}")
    print(f"total time wasted: {time.time() - current_time:.6f} 秒")

    # 可視化（僅當選擇的特徵數量為 2 時）
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

selected_columns = ['Feature1', 'Feature2']
print("selected_columns : ", selected_columns)
three_d_linear_regression(selected_columns)
#three_d_linear_regression(selected_columns, True)
print()


def fitness_function(features, norm=False):#input a particle contains all feature with 0 1
    if np.sum(features) == 0:  # 若無特徵被選中，返回較高的懲罰值
        return float('inf')  # 懲罰不選擇任何特徵的解

    # 根據粒子的選擇生成特徵名稱列表
    selected_columns = [f"Feature{i}" for i in range(1, 51) if features[i-1] == 1]
    
    # 如果選擇了特徵，則呼叫 three_d_linear_regression 計算 MSE
    if len(selected_columns) > 0:
        # 假設 three_d_linear_regression 會計算並返回 MSE
        mse = three_d_linear_regression(selected_columns)
        #print("test for now particle mse calculation: ", mse)
        return mse
    else:
        return float('inf')


# PSO实现
def pso_feature_selection(n, norm=False, max_iter=50, num_particles=100):
    np.random.seed(42)  # 设置随机种子
    num_features = 50  # 特征总数
    w_max, w_min = 0.9, 0.4  # 惯性权重范围
    c1, c2 = 1.5, 1.5  # 加速因子

    # 初始化粒子群
    positions = np.random.randint(0, 2, (num_particles, num_features))  # set 0 or 1 on particle 2d array[[0, 1, ....],...[1, 0,....]] particles x features (30x50)
    velocities = np.random.uniform(-1, 1, (num_particles, num_features))  # set velocties of particles
    personal_best_positions = positions.copy()  # local best position for each particles
    personal_best_scores = np.array([fitness_function(p) for p in positions])  # local best MSE, each p is like [0, 1, ....]
    print("test personaL_best_scores: ", personal_best_scores)
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]  # find the minimun MSE value from each local particle to be global position
    global_best_score = np.min(personal_best_scores)  # global best MSE from global position
    
    test_bestMSE = global_best_score
    test_best = global_best_position
    
    '''
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
            
            # 更新位置（用Sigmoid函数映射速度）
            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            positions[i] = np.where(np.random.rand(num_features) < sigmoid, 1, 0)
            
            # ensure feature count is n
            if np.sum(positions[i]) > n:
                indices = np.where(positions[i] == 1)[0]
                np.random.shuffle(indices)
                positions[i, indices[:np.sum(positions[i]) - n]] = 0
                print("position now: ", positions[i])
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

        print()
        print(f"iteration: {iteration + 1}/{max_iter} - global best MSE：{global_best_score}")
        print()

    # ensure global best solution has n features
    if np.sum(global_best_position) > n:
        indices = np.where(global_best_position == 1)[0]
        np.random.shuffle(indices)
        global_best_position[indices[:np.sum(global_best_position) - n]] = 0
    elif np.sum(global_best_position) < n:
        indices = np.where(global_best_position == 0)[0]
        np.random.shuffle(indices)
        global_best_position[indices[:n - np.sum(global_best_position)]] = 1

    print("global position for the 0 iteration: ", test_best)
    print("global best MSE for the 0 iteration: ", test_bestMSE)


    # 输出最终结果
    selected_features = [f"Feature{i}" for i in range(1, 51) if global_best_position[i-1] == 1]
    print('test global: ', global_best_position)
    print('test selected: ', selected_features)
    return selected_features, global_best_score
'''
# 主程序

if __name__ == "__main__":
    n = 5  # 选取特征数
    pso_feature_selection(n, norm=False)
    #selected_features, mse = pso_feature_selection(n, norm=False)
    #print(f"选择的特征：{selected_features}")
    #print(f"最优MSE：{mse}")

    selected_columns = ['Feature3', 'Feature44', 'Feature45', 'Feature46', 'Feature49']
    print("selected_columns : ", selected_columns)
    three_d_linear_regression(selected_columns)
    print()


'''
# 示例数据
np.random.seed(42)
X = pd.DataFrame(np.random.rand(100, 50), columns=[f'Feature_{i}' for i in range(50)])
y = np.random.rand(100)

# 1. 计算相关系数矩阵
correlation_with_y = X.corrwith(pd.Series(y))  # 特征与目标的相关性
correlation_matrix = X.corr()  # 特征之间的相关性

# 2. 特征重要性排序（按与目标的相关性绝对值降序）
sorted_features = correlation_with_y.abs().sort_values(ascending=False).index

# 3. 去冗余特征（逐步筛选相关性较低的特征）
def select_features(max_features, threshold=0.8):
    """
    选取指定数量的特征，同时确保特征间的相关性低于阈值。
    参数：
        max_features (int): 选择的特征数量。
        threshold (float): 特征间最大允许相关性。
    返回：
        list: 选择的特征名称。
    """
    selected_features = []
    
    for feature in sorted_features:
        # 检查当前特征与已选择特征的相关性是否低于阈值
        if all(abs(correlation_matrix[feature][sf]) < threshold for sf in selected_features):
            selected_features.append(feature)
        # 如果达到了最大特征数量，停止选择
        if len(selected_features) == max_features:
            break
            
    return selected_features

# 选择最佳特征
selected_features = select_features(max_features=10, threshold=0.7)  # 选择10个特征，特征间相关性<0.7

print("选择的特征：", selected_features)

# 4. 使用选择的特征构建线性回归模型
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model = LinearRegression()
selected_X = X[selected_features]

scores = cross_val_score(model, selected_X, y, cv=5, scoring='neg_mean_squared_error')
print("模型的均方误差：", -np.mean(scores))
'''