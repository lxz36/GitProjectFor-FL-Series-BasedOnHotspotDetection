import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
import torch.nn as nn

from model_pytorch import Model


def get_auc(tp_rates, fp_rates, logging_dir, round_num=None):
    if fp_rates[0] > fp_rates[-1]:
        tp_rates = tp_rates[::-1]
        fp_rates = fp_rates[::-1]
    else:
        tp_rates = tp_rates[:]
        fp_rates = fp_rates[:]
    if tp_rates[0] != 0.0 or fp_rates[0] != 0.0:
        tp_rates.insert(0, 0.0)
        fp_rates.insert(0, 0.0)
    if tp_rates[-1] != 1.0 or fp_rates[-1] != 1.0:
        tp_rates.append(1.0)
        fp_rates.append(1.0)
    auc_score = sklearn.metrics.auc(fp_rates, tp_rates)
    print("AUC: %.6f" % auc_score)

    plt.figure()
    plt.plot(fp_rates, tp_rates)
    plt.xlabel("FP rate")
    plt.ylabel("TP rate")
    plt.title("ROC Curve for prediction, AUC={0:.5f}".format(auc_score))
    plt.savefig(os.path.join(logging_dir,
                            'roc_curve-round-{}'.format(round_num or 'latest')))
    plt.close()


@torch.no_grad()
def get_group_norm(conv_module):
    weight = conv_module.weight.data.cpu().numpy()
    weight = np.transpose(weight, axes=[1, 0, 2, 3])
    weight = weight.reshape(weight.shape[0], -1)
    group_norm = np.linalg.norm(weight, axis=1)
    return group_norm


@torch.no_grad()
def fed_avg(clients_states, server_model, names_not_merge=[]):
    num_clients = len(clients_states)
    for name, m in server_model.named_modules():   #遍历每一层神经网络？
        if any(name.endswith(n) for n in names_not_merge):
            continue
        
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print('Merging layer:', name)
            m_data_avg = torch.zeros_like(m.weight.data, dtype=torch.float32)
            for c_state in clients_states:
                m_data_avg += c_state['model'][name+'.weight']
            m_data_avg.div_(num_clients)
            m.weight.data.copy_(m_data_avg)
            if m.bias is not None:
                m_data_avg = torch.zeros_like(m.bias.data, dtype=torch.float32)
                for c_state in clients_states:
                    m_data_avg += c_state['model'][name+'.bias']
                m_data_avg.div_(num_clients)
                m.bias.data.copy_(m_data_avg)

#定义带噪声的FedAvg
@torch.no_grad()
def fed_avgWithNoise(clients_states, server_model, names_not_merge=[], isFixedNoise=True,gradient=0,noiseVariance=0.0001):  #isFixedNoise代表是否加入固定噪声,gradient代表目前的梯度大小
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #给加上的噪声安排cuda计算

    num_clients = len(clients_states)

    for name, m in server_model.named_modules():  # 遍历每一层神经网络
        if any(name.endswith(n) for n in names_not_merge):
            continue

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print('Merging layer:', name)
            m_data_avg = torch.zeros_like(m.weight.data, dtype=torch.float32)
            for c_state in clients_states:
                m_data_avg += c_state['model'][name + '.weight']
                #可以选择加多次噪声or选择最终加一次噪声
            m_data_avg.div_(num_clients)

            #最终加一次噪声，加噪声在这里
            # print("加入噪声前：", m_data_avg[0][0][0])
            # 产生噪声
            if noiseVariance!=0:
                noiseValue = np.random.laplace(0, noiseVariance, list(m_data_avg.shape))  # list(n[0].shape)  将torch的维度转化为list的维度 // np.random.laplace(0, 0.0001, 1) 是 最小刻度的噪声，
                # print("加入的噪声：", torch.Tensor(noiseValue)[0][0][0])
                m_data_avg = m_data_avg + torch.Tensor(noiseValue).to(device)  # noiseValue就是加的对应形状的噪声
                # print("加入噪声后：", m_data_avg[0][0][0])

            m.weight.data.copy_(m_data_avg)
            if m.bias is not None:
                m_data_avg = torch.zeros_like(m.bias.data, dtype=torch.float32)
                for c_state in clients_states:
                    m_data_avg += c_state['model'][name + '.bias']
                m_data_avg.div_(num_clients)
                m.bias.data.copy_(m_data_avg)

#定义带噪声的FedAvg统计时间
@torch.no_grad()
def fed_avgWithNoiseAndTime(clients_states, server_model, names_not_merge=[], isFixedNoise=True,gradient=0,noiseVariance=0.0001):  #isFixedNoise代表是否加入固定噪声,gradient代表目前的梯度大小
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #给加上的噪声安排cuda计算

    num_clients = len(clients_states)

    t_c1,t_c2=0,0 #初始统计时间

    for name, m in server_model.named_modules():  # 遍历每一层神经网络
        if any(name.endswith(n) for n in names_not_merge):
            continue

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print('Merging layer:', name)
            m_data_avg = torch.zeros_like(m.weight.data, dtype=torch.float32)
            for c_state in clients_states:
                m_data_avg += c_state['model'][name + '.weight']
                #可以选择加多次噪声or选择最终加一次噪声
            m_data_avg.div_(num_clients)


            #最终加一次噪声，加噪声在这里
            # print("加入噪声前：", m_data_avg[0][0][0])
            # 产生噪声
            # if noiseVariance!=0:
            #     noiseValue = np.random.laplace(0, noiseVariance, list(m_data_avg.shape))  # list(n[0].shape)  将torch的维度转化为list的维度 // np.random.laplace(0, 0.0001, 1) 是 最小刻度的噪声，
            #     # print("加入的噪声：", torch.Tensor(noiseValue)[0][0][0])
            #     m_data_avg = m_data_avg + torch.Tensor(noiseValue).to(device)  # noiseValue就是加的对应形状的噪声
                # print("加入噪声后：", m_data_avg[0][0][0])


            ################################时间测试
            # 方法1  时间单位是ms
            start = torch.cuda.Event(enable_timing=True)  # 设置一个开始的事件
            end = torch.cuda.Event(enable_timing=True)  # 设置一个结束的事件
            start.record(stream=torch.cuda.current_stream())  # 开始记录

            if noiseVariance!=0:
                noiseValue = np.random.laplace(0, noiseVariance, list(m_data_avg.shape))  # list(n[0].shape)  将torch的维度转化为list的维度 // np.random.laplace(0, 0.0001, 1) 是 最小刻度的噪声，
                # print("加入的噪声：", torch.Tensor(noiseValue)[0][0][0])
                m_data_avg = m_data_avg + torch.Tensor(noiseValue).to(device)  # noiseValue就是加的对应形状的噪声


            end.record(stream=torch.cuda.current_stream())  # 结束记录
            end.synchronize()  # 同步，保证时间准确
            t_c1 += start.elapsed_time(end)  # 计算时间差
            print("遍历神经网络的时间差(ms)：", start.elapsed_time(end))


            # 方法2  时间单位是s
            start = time.time()  # 开始记录

            # 执行的代码
            if noiseVariance!=0:
                noiseValue = np.random.laplace(0, noiseVariance, list(m_data_avg.shape))  # list(n[0].shape)  将torch的维度转化为list的维度 // np.random.laplace(0, 0.0001, 1) 是 最小刻度的噪声，
                # print("加入的噪声：", torch.Tensor(noiseValue)[0][0][0])
                m_data_avg = m_data_avg + torch.Tensor(noiseValue).to(device)  # noiseValue就是加的对应形状的噪声

            torch.cuda.synchronize()
            end = time.time()  # 结束记录
            t_c2 += (end - start)  # 计算时间差

            print("遍历神经网络的时间差(s)：", end - start)
            
            




            m.weight.data.copy_(m_data_avg)
            if m.bias is not None:
                m_data_avg = torch.zeros_like(m.bias.data, dtype=torch.float32)
                for c_state in clients_states:
                    m_data_avg += c_state['model'][name + '.bias']
                m_data_avg.div_(num_clients)
                m.bias.data.copy_(m_data_avg)
    print("加入噪声时需要的时间(ms)：",t_c1)
    print("加入噪声时需要的时间2(s)：", t_c2)


@torch.no_grad()
def restore_from_server(client_model, server_model, names_not_restore=[]):
    server_dict = server_model.state_dict()
    for name, m in client_model.named_modules():
        if any(name.endswith(n) for n in names_not_restore):
            continue
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print("Restoring layer", name)
            m.weight.data.copy_(server_dict[name+'.weight'])
            if m.bias is not None:
                m.bias.data.copy_(server_dict[name+'.bias'])


def soft_cross_entropy(logits, targets):
    '''
    returns cross entropy loss that allows for float inputs (soft labels)
    logits & targets are of the same one-hot shape
    '''
    log_probs = nn.functional.log_softmax(logits, dim=1)
    return - (targets * log_probs).mean()


def l2_reg_but_final_fc(model, final_fc_name='final_fc'):
    reg = None
    for name, m in model.named_modules():
        if name == final_fc_name:
            continue
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):            
            if reg is None:
                reg = m.weight.pow(2).sum()
            else:
                reg += m.weight.pow(2).sum()
            if m.bias is not None:
                reg += m.bias.pow(2).sum()
    return reg


def l2_fedprox(model, server_model):
    reg = None
    for m, ms in zip(model.modules(), server_model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            assert m.weight.size() == ms.weight.size()
            if reg is None:
                reg = (m.weight - ms.weight).pow(2).sum()
            else:
                reg += (m.weight - ms.weight).pow(2).sum()
            if m.bias is not None:
                assert m.bias.size() == ms.bias.size()
                reg += (m.bias - ms.bias).pow(2).sum()
    return reg


def group_lasso_channel_wise(conv_module):
    weight = torch.transpose(conv_module.weight, 0, 1)
    weight = weight.reshape(weight.shape[0], -1)
    penalty = torch.linalg.norm(weight, dim=1).sum()
    return penalty


'''
Train the model max_iter steps
'''
def train(model, data_loader, optimizer, criterion, batch_size, l2_reg_factor, group_lasso_strength,
          max_itr, prev_steps, data_size, display_step, device,
          untrained_modules=[]):
    # fixed requires_grad of untrained modules
    for name, m in model.named_modules():
        if name in untrained_modules:
            print('Not training', name)
            for p in m.parameters():
                p.requires_grad = False

    # Display learning rate
    param_groups = optimizer.state_dict()['param_groups']
    print('LR={0:.5f}, len={1}'.format(param_groups[-1]['lr'], len(param_groups)))
          
    # start training of this client for 1 round (given #steps)
    train_loss = train_l2_reg = 0
    l2_norm_but_final_fc = l2_norm_final_fc = 0
    this_round_step = 0
    correct = total = 0
    start = time.time()
    while True:
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # TODO: support biased learning
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            l2_reg = l2_reg_but_final_fc(model)
            lasso_reg = group_lasso_channel_wise(model.conv1_1[0])

            total_loss = loss + l2_reg_factor * l2_reg + group_lasso_strength * lasso_reg
            total_loss.backward()
            optimizer.step()

            # TODO: check L2 norm anyways... (detailed)
            with torch.no_grad():
                train_loss += loss.item()
                train_l2_reg += l2_reg_factor * l2_reg.item()
                l2_norm_but_final_fc += l2_reg.sqrt().item()
                l2_norm_final_fc += model.final_fc.weight.norm().item()
                prediction = outputs.argmax(dim=1)
                ground_truth = targets.argmax(dim=1)
                this_total = targets.size(0)
                this_correct = prediction.eq(ground_truth).sum().item()
                total += this_total
                correct += this_correct

            this_round_step += 1

            if this_round_step % display_step == 0 or this_round_step == 1:
                # display brief
                end = time.time() 
                _step = prev_steps + this_round_step
                num_samples_per_sec = display_step * batch_size / (end - start)
                print('[Step={0:4d} Epoch={1:4.1f}] | '
                    .format(_step, _step * batch_size / data_size)
                    + 'Loss={0:.5f} | '.format(train_loss / this_round_step)
                    + 'Reg={0:.5f} | '.format(train_l2_reg / this_round_step)
                    + 'acc={0:.4f} | '.format(this_correct / this_total)
                    + 'L2-Norm={0:.3f} | '.format(l2_norm_but_final_fc / this_round_step)
                    + 'L2-Norm(final)={0:.3f} | '.format(l2_norm_final_fc / this_round_step)
                    + '{0:.1f} samples/s | {1:4.1f} steps/s'
                    .format(num_samples_per_sec, num_samples_per_sec / batch_size))
                start = time.time()

            if this_round_step >= max_itr:
                break
        if this_round_step >= max_itr:
            break
    assert this_round_step == max_itr, 'Error in round step number calculation'

    # restore requires_grad of untrained modules
    for name, m in model.named_modules():
        if name in untrained_modules:
            for p in m.parameters():
                p.requires_grad = True


'''
Train the model max_iter steps
'''
def train_fedprox(model, server_model, data_loader, optimizer, criterion, batch_size, l2_reg_factor,
        fedprox_mu, max_itr, prev_steps, data_size, display_step, device, untrained_modules=[]):
    # fixed requires_grad of untrained modules
    for name, m in model.named_modules():
        if name in untrained_modules:
            print('Not training', name)
            for p in m.parameters():
                p.requires_grad = False

    # Display learning rate
    param_groups = optimizer.state_dict()['param_groups']
    print('LR={0:.5f}, len={1}'.format(
        param_groups[-1]['lr'], len(param_groups)))

    # start training of this client for 1 round (given #steps)
    train_loss = train_l2_reg = 0
    l2_norm_but_final_fc = l2_norm_final_fc = 0
    train_fedprox_reg = 0
    this_round_step = 0
    correct = total = 0
    start = time.time()
    while True:
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # TODO: support biased learning
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            l2_reg = l2_reg_but_final_fc(model)
            fedprox_reg = l2_fedprox(model, server_model)

            total_loss = loss + l2_reg_factor * l2_reg + fedprox_mu * fedprox_reg
            total_loss.backward()
            optimizer.step()

            # TODO: check L2 norm anyways... (detailed)
            with torch.no_grad():
                train_loss += loss.item()
                train_l2_reg += l2_reg_factor * l2_reg.item()
                l2_norm_but_final_fc += l2_reg.sqrt().item()
                l2_norm_final_fc += model.final_fc.weight.norm().item()
                train_fedprox_reg += fedprox_mu * fedprox_reg.item()
                prediction = outputs.argmax(dim=1)
                ground_truth = targets.argmax(dim=1)
                this_total = targets.size(0)
                this_correct = prediction.eq(ground_truth).sum().item()
                total += this_total
                correct += this_correct

            this_round_step += 1

            if this_round_step % display_step == 0 or this_round_step == 1:
                # display brief
                end = time.time()
                _step = prev_steps + this_round_step
                num_samples_per_sec = display_step * batch_size / (end - start)
                print('[Step={0:4d} Epoch={1:4.1f}] | '
                      .format(_step, _step * batch_size / data_size)
                      + 'Loss={0:.5f} | '.format(train_loss / this_round_step)
                      + 'Reg={0:.5f} | '.format(train_l2_reg / this_round_step)
                      + 'FP-Reg={0:.3f} | '.format(train_fedprox_reg / this_round_step)
                      + 'acc={0:.4f} | '.format(this_correct / this_total)
                      + 'L2-Norm={0:.3f} | '.format(l2_norm_but_final_fc / this_round_step)
                      + 'L2-Norm(final)={0:.3f} | '.format(l2_norm_final_fc / this_round_step)
                      + '{0:.1f} samples/s | {1:4.1f} steps/s'
                      .format(num_samples_per_sec, num_samples_per_sec / batch_size))
                start = time.time()

            if this_round_step >= max_itr:
                break
        if this_round_step >= max_itr:
            break
    assert this_round_step == max_itr, 'Error in round step number calculation'

    # restore requires_grad of untrained modules
    for name, m in model.named_modules():
        if name in untrained_modules:
            for p in m.parameters():
                p.requires_grad = True


'''
Test model on a given testing dataset
'''
@torch.no_grad()
def test(model, data_loader, criterion, batch_size, display_step, device):
    test_step = 0
    test_loss = 0
    correct = total = 0
    total_pos = total_neg = 0
    true_pos_p = false_pos_p = 0

    model.eval()
    start = time.time()
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        prediction = outputs.argmax(dim=1)
        ground_truth = targets.argmax(dim=1)
        result = prediction.eq(ground_truth)

        total += targets.size(0)
        correct += result.sum().item()

        total_pos += ground_truth.eq(1).sum().item()
        total_neg += ground_truth.eq(0).sum().item()
        true_pos_p += result[ground_truth.eq(1)].sum().item()
        false_pos_p += result.eq(0)[ground_truth.eq(0)].sum().item()

        test_step += 1

        if test_step % display_step == 0:
            # display brief
            end = time.time()
            num_samples_per_sec = display_step * batch_size / (end - start)
    #         print(
    #             '[Step={0:4d}] | Loss={1:.5f} | acc={2:.4f} | tpr={3:.4f} | fpr={4:.4f} | {5:.1f} samples/s | {6:4.1f} steps/s'
    #             .format(test_step,
    #                     test_loss / test_step,
    #                     correct / total,
    #                     true_pos_p / total_pos,
    #                     false_pos_p / total_neg,
    #                     num_samples_per_sec,
    #                     num_samples_per_sec / batch_size))
    #         start = time.time()
    # # testing on current test dataset over
    # print('Avg test loss: {0:.5f}, Avg test acc: {1:.5f}, Avg tpr: {2:.5f}, Avg fpr: {3:.5f}, total FA: {4:d}'
    #         .format(test_loss / test_step, correct / total, true_pos_p / total_pos, false_pos_p / total_neg, false_pos_p))

    #这5个数据分别是Avg test loss， acc,tpr,fpr, FA
    outSum=[test_loss / test_step, correct / total, true_pos_p / total_pos, false_pos_p / total_neg, false_pos_p]
    return outSum
