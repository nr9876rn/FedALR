import numpy as np
import cvxpy as cp
import copy

import torch


def multiply_matrices(H, S):
    n = H.shape[0]
    matrix_shape = S[0].shape
    result = []

    for i in range(n):
        current_result = np.zeros(matrix_shape, dtype=object)
        for j in range(n):
            current_result += H[i, j] * S[j]
        result.append(current_result)

    return np.array(result)


def compute_cos_matrix(args, model, initial_global_parameters, dw):
    # if args.dataset == 'yahoo':
    #     model_temp = {k: model[k] for k in range(len(model))}
    #     model_similarity_matrix = torch.zeros((len(model_temp), len(model_temp)))
    #     index_clientid = list(model_temp.keys())
    #
    #     # 过滤掉 Embedding 层的参数
    #     def filter_embedding_params(params):
    #         return {k: v for k, v in params.items() if 'embedding' not in k}
    #
    #     # # 过滤掉 Embedding 层和 convs 层的参数
    #     # def filter_embedding_params(params):
    #     #     return {k: v for k, v in params.items() if 'embedding' not in k and 'convs' not in k}
    #
    #     for i in range(len(model_temp)):
    #         model_i = model_temp[index_clientid[i]].state_dict()
    #         # 过滤掉 Embedding 层的参数
    #         model_i_filtered = filter_embedding_params(model_i)
    #         initial_global_parameters_filtered = filter_embedding_params(initial_global_parameters)
    #         for key in dw[index_clientid[i]]:
    #             if key in model_i_filtered and key in initial_global_parameters_filtered:
    #                 dw[index_clientid[i]][key] = model_i_filtered[key] - initial_global_parameters_filtered[key]
    #
    #     for i in range(len(model_temp)):
    #         for j in range(i, len(model_temp)):
    #             # 过滤掉 Embedding 层的参数后计算相似度
    #             dw_i_filtered = filter_embedding_params(dw[index_clientid[i]])
    #             dw_j_filtered = filter_embedding_params(dw[index_clientid[j]])
    #             diff = torch.nn.functional.cosine_similarity(weight_flatten_all(dw_i_filtered).unsqueeze(0),
    #                                                          weight_flatten_all(dw_j_filtered).unsqueeze(0))
    #             model_similarity_matrix[i, j] = diff
    #             model_similarity_matrix[j, i] = diff
    # else:
    model_temp = {k: model[k] for k in range(len(model))}
    model_similarity_matrix = torch.zeros((len(model_temp),len(model_temp)))
    index_clientid = list(model_temp.keys())
    for i in range(len(model_temp)):
        model_i = model_temp[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]
    for i in range(len(model_temp)):
        for j in range(i, len(model_temp)):
            diff = torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
            model_similarity_matrix[i, j] = diff
            model_similarity_matrix[j, i] = diff

    return model_similarity_matrix


def compute_dis_matrix(args, model):
    model_temp = {k: model[k] for k in range(len(model))}
    model_distance_matrix = torch.zeros((len(model_temp), len(model_temp)))  # 存储欧氏距离
    index_clientid = list(model_temp.keys())
    # 计算欧氏距离矩阵
    for i in range(len(model_temp)):
        for j in range(i, len(model_temp)):
            # 计算两个模型参数差异之间的欧氏距离
            diff = torch.norm(weight_flatten_all(model_temp[i].state_dict()) - weight_flatten_all(model_temp[j].state_dict()))
            model_distance_matrix[i, j] = diff
            model_distance_matrix[j, i] = diff  # 对称矩阵
    return model_distance_matrix


def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_part(model):
    params = []
    for k in model:
        if 'classifier' in k:
        # if 'fc' in k:
        # if 'conv' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params




def optimizing_weight_matrix(cos_matrix, p_vector, alpha, js_matrix):
    w_matrix = []
    n = cos_matrix.shape[0]
    P = cp.atoms.affine.wraps.psd_wrap(np.identity(n))
    ones_matrix = np.identity(n)
    zero_vector = np.zeros(n)
    ones_vector = np.ones((1, n))
    ones = np.ones(1)
    #cv = cumputer_cv_vector(cos_matrix)
    # cv_res = []
    for i in range(n):
        cos_vector = copy.deepcopy(cos_matrix[i])
        js_matrix = torch.tensor(np.array(js_matrix))
        js_vector = copy.deepcopy(js_matrix[i])

        # =max
        temp = copy.deepcopy(cos_vector)
        before_i = temp[:i]
        after_i = temp[i + 1:]
        temp = torch.cat((before_i, after_i))
        cos_vector[i] = max(temp)

        temp = copy.deepcopy(js_vector)
        before_i = temp[:i]
        after_i = temp[i + 1:]
        temp = torch.cat((before_i, after_i))
        js_vector[i] = max(temp)

        cv = compute_cv(cos_vector, i)
        if cv <= 0.05:
            cv = 0
        print('node:', i, cv)
        # cv_res.append(cv)

        # adapt
        # temp = copy.deepcopy(cos_vector)
        # before_i = temp[:i]
        # after_i = temp[i + 1:]
        # temp = torch.cat((before_i, after_i))
        # cos_vector[i] = min(1, max(temp) + cv*(1-max(temp)))
        #print('cos_vector:', cos_vector)
        # cv = 1
        x = cp.Variable(n)
        # temp = alpha * cos_vector * cv + alpha * p_vector
        temp = alpha * js_vector + alpha * p_vector
        # temp = cos_vector + 0.8 * p_vector
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - temp.T @ x),
                          [ones_matrix @ x >= zero_vector,
                           ones_vector @ x == ones]
                          )
        prob.solve()
        w_matrix.append(x.value)
    # print('cv:', cv_res)
    return w_matrix


def compute_cv(cos_vector, node):
    temp = copy.deepcopy(cos_vector)
    temp[node] = np.nan
    return np.nanstd(temp) / (np.nanmean(temp) + 0.1)

def cumputer_cv_vector(matrix):
    temp = copy.deepcopy(matrix)
    for i in range(20):
        temp[i][i] = np.nan
    cv = np.nanstd(temp, axis=1) / (np.nanmean(temp, axis=1) + 0.1)
    print('cv:', cv)
    result = np.nanmean(cv)
    print('mean_cv:', result)
    return result

def compute_js(traindata_cls_counts, classes_size):
    n = traindata_cls_counts.astype(int).shape[0]
    data_info = traindata_cls_counts.astype(int)
    temp = np.zeros((n, classes_size))
    for i in range(n):
        temp[i] = [x / np.sum(data_info[i]) for x in data_info[i]]
    js_matrix = calculate_js_divergence_matrix(temp)  # 1-JS
    print('1-JS散度:', js_matrix)
    return js_matrix







def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)

def kl_divergence(p, q):
    # 确保概率分布的元素非负
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # 避免除零错误
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    return np.sum(p * np.log2(p / q))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 1-0.5 * (kl_divergence(p, m) + kl_divergence(q, m))  # 1-JS


def calculate_js_divergence_matrix(data):
    num_clients = data.shape[0]
    divergence_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            # 计算每个客户端标签分布的概率
            p = data[i] / np.sum(data[i])
            q = data[j] / np.sum(data[j])
            divergence_matrix[i, j] = js_divergence(p, q)
    return divergence_matrix