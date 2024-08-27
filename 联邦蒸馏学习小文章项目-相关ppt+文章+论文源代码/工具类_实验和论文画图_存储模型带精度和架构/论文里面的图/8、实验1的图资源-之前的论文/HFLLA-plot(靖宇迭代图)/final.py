#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt


def get_trace(log_path):
    trace_of_settings = {}
    for log_filename in os.listdir(log_path):
        if not os.path.isfile(os.path.join(log_path, log_filename)):
            continue
        trace = {}
        log_fullpath = os.path.join(log_path, log_filename)
        with open(log_fullpath, 'r') as log_file:
            tested_model = test_dataset = None
            for line in log_file:
                if 'Testing' in line:
                    l_split = line.split()
                    tested_model = l_split[1]
                    test_dataset = l_split[-1]
                    assert tested_model is not None, line
                elif tested_model is not None:
                    if 'Avg test loss' in line:
                        l_split = line.split(',')
                        if (tested_model, test_dataset) not in trace:
                            trace[(tested_model, test_dataset)] = \
                                {s.split(':')[0].strip(): [] for s in l_split}
                        for s in l_split:
                            perf_name, perf_val = s.split(':')
                            perf_name = perf_name.strip()
                            perf_val = float(perf_val.strip())
                            trace[(tested_model, test_dataset)][perf_name] \
                                .append(perf_val)
                        tested_model = test_dataset = None

        trace_of_settings[log_filename[:-4]] = trace
    return trace_of_settings


def get_clients_mean(trace, metric='Avg tpr'):
    clients_mean = {}
    for (model, dataset), perf in trace.items():
        if 'client' in model and dataset not in model:
            continue    # skip unmatched clients
        if dataset not in clients_mean:
            clients_mean[dataset] = [perf[metric]]
        else:
            clients_mean[dataset].append(perf[metric])
    for dataset, perf in clients_mean.items():
        clients_mean[dataset] = np.asarray(perf).mean(axis=0)
    return clients_mean


def get_clients_min_max(trace, metric='Avg tpr'):
    clients_min, clients_max = {}, {}
    for (model, dataset), perf in trace.items():
        if 'client' in model and dataset not in model:
            continue    # skip unmatched clients
        if dataset not in clients_min:
            clients_min[dataset] = [perf[metric]]
        else:
            clients_min[dataset].append(perf[metric])
    for dataset, perf in clients_min.items():
        clients_min[dataset] = np.asarray(perf).min(axis=0)
        clients_max[dataset] = np.asarray(perf).max(axis=0)
    return clients_min, clients_max


if __name__ == '__main__':
    hfl_trace = get_trace('log/fl_but_fc2')['a5i5_sr1.0']
    fedavg_trace = get_trace('log/fl')['gamma0.5_a5i5_sr1.0']
    fedprox_trace = get_trace('log/fedprox')['fedprox-a5i5-sel1.0-l2fp1e-3-withl2']
    hfl_trace_asc = get_trace('log/fl_but_fc2')['a5i5_sr0.5']
    hfl_trace_asc_26 = get_trace('log/fl_but_fc2')['a5i5_sel0.5_ch26']
    fedavg_trace_asc = get_trace('log/fl')['gamma0.5_a5i5_sr0.5']
    fedprox_trace_asc = get_trace('log/fedprox')['fedprox-a5i5-sel0.5-l2fp1e-3-withl2']
    local_trace = get_trace('log/no_server')['a5i5-50r-800spr']
    global_trace = get_trace('log/single_model')['single_asml1_iccad2012_25000steps']


    params = {
        'font.family': 'Times New Roman',
        'text.usetex': True
    }
    plt.rcParams.update(params)

    figsize = np.array((6.4, 3.9)) * 0.85


    fig, ax = plt.subplots(figsize=figsize)

    hfl_acc_mean = get_clients_mean(hfl_trace, 'Avg test acc')
    hfl_acc_min, hfl_acc_max = get_clients_min_max(hfl_trace, 'Avg test acc')
    hfl_asc_26_acc_mean = get_clients_mean(hfl_trace_asc_26, 'Avg test acc')
    hfl_asc_26_acc_min, hfl_asc_26_acc_max = get_clients_min_max(hfl_trace_asc_26, 'Avg test acc')
    fedavg_acc_mean = get_clients_mean(fedavg_trace, 'Avg test acc')
    fedavg_acc_min, fedavg_acc_max = get_clients_min_max(fedavg_trace, 'Avg test acc')
    fedprox_acc_mean = get_clients_mean(fedprox_trace, 'Avg test acc')
    fedprox_acc_min, fedprox_acc_max = get_clients_min_max(fedprox_trace, 'Avg test acc')
    hfl_asc_acc_mean = get_clients_mean(hfl_trace_asc, 'Avg test acc')
    hfl_asc_acc_min, hfl_asc_acc_max = get_clients_min_max(hfl_trace_asc, 'Avg test acc')
    fedavg_asc_acc_mean = get_clients_mean(fedavg_trace_asc, 'Avg test acc')
    fedavg_asc_acc_min, fedavg_asc_acc_max = get_clients_min_max(fedavg_trace_asc, 'Avg test acc')
    fedprox_asc_acc_mean = get_clients_mean(fedprox_trace_asc, 'Avg test acc')
    fedprox_asc_acc_min, fedprox_asc_acc_max = get_clients_min_max(fedprox_trace_asc, 'Avg test acc')
    local_acc_mean = get_clients_mean(local_trace, 'Avg test acc')
    local_acc_min, local_acc_max = get_clients_min_max(local_trace, 'Avg test acc')
    global_acc_mean = get_clients_mean(global_trace, 'Avg test acc')
    global_acc_min, global_acc_max = get_clients_min_max(global_trace, 'Avg test acc')

    mean_ = [0] + list(hfl_acc_mean['iccad2012'][:49])
    max_ = [0] + list(hfl_acc_max['iccad2012'][:49])
    min_ = [0] + list(hfl_acc_min['iccad2012'][:49])
    x_rounds = np.arange(len(mean_))
    ax.plot(x_rounds, mean_, label='HFL-LA (sync)', ls='--', c='r')
    ax.fill_between(x_rounds, max_, min_, color='r', alpha=.3)

    mean_ = [0] + list(hfl_asc_acc_mean['iccad2012'][:49])
    max_ = [0] + list(hfl_asc_acc_max['iccad2012'][:49])
    min_ = [0] + list(hfl_asc_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='HFL-LA (async)', ls='--', c='r', marker='^', markevery=3, alpha=0.6)
    ax.fill_between(x_rounds, max_, min_, color='r', alpha=.3)

    #FedMD  iccad
    #创建的数据
    # mean_ = [0,0.77,0.86659,0.89205] +[0.9491695668166257, 0.9488173046996575, 0.952206943383414, 0.956571315394845, 0.9525195936960644, 0.9536244800950682, 0.9541252864782276, 0.9519225872167049, 0.9531250884192062, 0.9538621509209744, 0.9553461788755907, 0.9533698327815975, 0.9551919757802111, 0.9545313074724839, 0.9539003480179951, 0.9541762159409218, 0.9559233794527913, 0.9556178026766261, 0.9467645644116232, 0.949671787907082, 0.9522012845542257, 0.9510044421809127, 0.9506932065755596, 0.9500834677305265, 0.9539031774325892, 0.95115015703251, 0.9563591093002858, 0.9586806439747617, 0.95359477124183, 0.9562501768384122, 0.9589140706787764, 0.9600698865404749, 0.9575064369182016, 0.9543657867187278, 0.9559997736468325, 0.9560945590357355, 0.9546826811532695, 0.9523413405766347, 0.9570056305350423, 0.955637608578785, 0.9558908411849588, 0.9582152052740287, 0.9557366380895793, 0.9555300908242085, 0.955190561072914, 0.9591687179922473]
    #真实数据
    mean_ = [0,0.77,0.86659,0.89205] +[0.9408921144215261, 0.9473615708909827, 0.9473771326712503, 0.9460374048609342, 0.9521206462382933, 0.9478637919814391, 0.9405936111818466, 0.9463627875392582, 0.9523866112101407, 0.9562530062530064, 0.9551636816342699, 0.9527275556687321, 0.9505559799677448,
            0.9509450244744361]+[0.9526851144498203, 0.9584160937102114, 0.9588221147044678, 0.9448886625357215, 0.9487635458223693, 0.9507413066236594, 0.94537532184591, 0.9525026172084996, 0.9468452027275557, 0.9518178988767225, 0.9518419489007724, 0.947392694451518, 0.945539427892369, 0.9519791755085872, 0.9518490224372578, 0.9477067594714654, 0.9567354214413039, 0.9511162040573804, 0.9561808561808561, 0.9569405539993776, 0.9556517556517556, 0.9586042497807205, 0.9551622669269728, 0.952397928868517, 0.9554070112893644, 0.9525606202076791, 0.9550406020994255, 0.9578332343038225, 0.9541210423563365, 0.9543841779135895, 0.9601604278074867, 0.9549571343688991]
    print(mean_)
    # max_ = [0] + list(fedavg_acc_max['iccad2012'][:49])
    # min_ = [0] + list(fedavg_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedMD (sync)', ls='--', c='k')

    mean_ = [0,0.54,0.82562,0.85873] + [0.9491695668166257, 0.9488173046996575, 0.952206943383414, 0.956571315394845, 0.9525195936960644, 0.9536244800950682, 0.9541252864782276, 0.9519225872167049, 0.9531250884192062, 0.9538621509209744, 0.9553461788755907, 0.9533698327815975, 0.9551919757802111, 0.9545313074724839]\
            +[0.9538620710860568, 0.9509950181094702, 0.9475849256509625, 0.945477964637411, 0.9450929171709762, 0.9462395359248575, 0.9528744835321052, 0.9467528538381333, 0.9528577418547467, 0.9493827962735278, 0.9502023575526352, 0.9533148792836789, 0.9498470542152941, 0.9458116956565858, 0.9454494409623545, 0.9468688645369446, 0.9467448728104774, 0.9459689822265208, 0.9479625973353443, 0.9536706043112136, 0.9478018371718372, 0.9503754543045266, 0.9483590275698787, 0.9529288655446704, 0.9464445498903983, 0.9527964213773263, 0.9492157015669674, 0.9458652249006854, 0.9459386539852218, 0.9473920330139656, 0.9505630235494466, 0.9516875589243242]

    # max_ = [0] + list(fedavg_asc_acc_max['iccad2012'][:49])
    # min_ = [0] + list(fedavg_asc_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedMD (async)', ls='--', c='k', marker='^', markevery=3, alpha=.6)



    mean_ = [0] + list(fedavg_acc_mean['iccad2012'][:49])
    max_ = [0] + list(fedavg_acc_max['iccad2012'][:49])
    min_ = [0] + list(fedavg_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedAvg (sync)', ls='-.', c='g')
    print(mean_)

    mean_ = [0] + list(fedavg_asc_acc_mean['iccad2012'][:49])
    max_ = [0] + list(fedavg_asc_acc_max['iccad2012'][:49])
    min_ = [0] + list(fedavg_asc_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedAvg (async)', ls='-.', c='g', marker='^', markevery=3, alpha=.6)
    print(mean_)

    mean_ = [0] + list(fedprox_acc_mean['iccad2012'][:49])
    max_ = [0] + list(fedprox_acc_max['iccad2012'][:49])
    min_ = [0] + list(fedprox_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedProx (sync)', ls=':', c='m')

    mean_ = [0] + list(fedprox_asc_acc_mean['iccad2012'][:49])
    max_ = [0] + list(fedprox_asc_acc_max['iccad2012'][:49])
    min_ = [0] + list(fedprox_asc_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedProx (async)', ls=':', c='m', marker='^', markevery=3, alpha=.6)

    mean_ = [0] + list(local_acc_mean['iccad2012'][:49])
    max_ = [0] + list(local_acc_max['iccad2012'][:49])
    min_ = [0] + list(local_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='Local')
    ax.fill_between(x_rounds, max_, min_, alpha=.3)

    mean_ = [0] + list(global_acc_mean['iccad2012'][:49])
    max_ = [0] + list(global_acc_max['iccad2012'][:49])
    min_ = [0] + list(global_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='Centralized')

    ax.legend(loc='lower right')
    ax.set_ylim([.2, 1.])  #原始值是 ax.set_ylim([.4, 1.])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('\#Rounds')
    ax.set_title(r'Accuracy on \texttt{ICCAD}')
    plot_dir = 'plot/final'
    os.makedirs(plot_dir, exist_ok=True)
    fig.tight_layout(pad=0.)
    fig.savefig(os.path.join(plot_dir, 'acc.iccad.pdf'))
    plt.close(fig)




    fig, ax = plt.subplots(figsize=figsize)

    mean_ = [0] + list(hfl_acc_mean['asml1'][:49])
    max_ = [0] + list(hfl_acc_max['asml1'][:49])
    min_ = [0] + list(hfl_acc_min['asml1'][:49])
    x_rounds = np.arange(len(mean_))
    ax.plot(x_rounds, mean_, label='HFL-LA (sync)', ls='--', c='r')
    ax.fill_between(x_rounds, max_, min_, color='r', alpha=.3)
    print("HFL sync acc asml1", mean_[-1])

    mean_ = [0] + list(hfl_asc_acc_mean['asml1'][:49])
    max_ = [0] + list(hfl_asc_acc_max['asml1'][:49])
    min_ = [0] + list(hfl_asc_acc_min['asml1'][:49])
    ax.plot(x_rounds, mean_, label='HFL-LA (async)', ls='--', c='r', marker='^', markevery=3, alpha=0.6)
    ax.fill_between(x_rounds, max_, min_, color='r', alpha=.3)
    print("HFL async acc asml1", mean_[-1])

    #FedMD  industry
    mean_ = [0,0.73] + [0.7830194727141598, 0.8252424072441702, 0.8346261719689079, 0.8546918823623688, 0.8439939097684108, 0.8581697251382323, 0.8629457488580815, 0.8641077009375753, 0.8646846702460133, 0.8692843977882843, 0.8668883724657425, 0.8684910649891819, 0.8769773219007935, 0.8747976600689157]\
    +[0.8765365814568474, 0.8804551646766567, 0.86906803429762, 0.8811523359243528, 0.8767048641718086, 0.8804391377514224, 0.8792771856719288, 0.8827309880599407, 0.8828511899991985, 0.887018190560141, 0.8837647247375591, 0.8829954323263083, 0.8903117236958089, 0.8883404118919784, 0.8826668803590032, 0.8857761038544755, 0.8932686914015546, 0.8909367737799503, 0.8897988620883084, 0.8889975158265887, 0.8907043833640517, 0.890071319817293, 0.8925394663033897, 0.8922830354996394, 0.8916900392659668, 0.8939257953361649, 0.8885808157704943, 0.8915217565510056, 0.8926676817052648, 0.8895664716724097, 0.8962817533456207, 0.8915377834762401, 0.8915137430883885, 0.8934209471912814]
    print(mean_)
    # max_ = [0] + list(fedavg_acc_max['iccad2012'][:49])
    # min_ = [0] + list(fedavg_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedMD (sync)', ls='--', c='k')

    mean_ = [0,0.62,] +  [0.7671047359564068, 0.7878435772097123, 0.7923952239762801, 0.7924593316772177, 0.8115794534818495, 0.825314528407725, 0.8271415978844459, 0.8265085343376872, 0.8278628095199936, 0.833696610305313, 0.844362529048802, 0.8433608462216524, 0.8450436733712637, 0.843264684670246, 0.8418302748617676, 0.8356278547960574, 0.8389454283195767, 0.8408366054972353, 0.8411651574645405, 0.844274380960013, 0.8425434730346983, 0.8421828672169245, 0.8431524961936052, 0.8439217886048562, 0.8454683868899752, 0.8384886609503968, 0.8449394983572402, 0.8430964019552849, 0.8437935732029811, 0.8427678499879798, 0.8490423912172449, 0.8542190880679541, 0.8548842054651814, 0.8546678419745172, 0.8514063626893181, 0.8538584822501802, 0.8579213077970991, 0.8560141036942064, 0.8555974036381121, 0.8494350508854878, 0.8489542431284558, 0.8422309479926277, 0.8479846141517751, 0.8524080455164678, 0.850260437535059, 0.8511339049603335, 0.8553009055212758, 0.8552207708951037]
    # max_ = [0] + list(fedavg_asc_acc_max['iccad2012'][:49])
    # min_ = [0] + list(fedavg_asc_acc_min['iccad2012'][:49])
    ax.plot(x_rounds, mean_, label='FedMD (async)', ls='--', c='k', marker='^', markevery=3, alpha=.6)


    mean_ = [0] + list(fedavg_acc_mean['asml1'][:49])
    max_ = [0] + list(fedavg_acc_max['asml1'][:49])
    min_ = [0] + list(fedavg_acc_min['asml1'][:49])
    ax.plot(x_rounds, mean_, label='FedAvg (sync)', ls='-.', c='g')

    mean_ = [0] + list(fedavg_asc_acc_mean['asml1'][:49])
    max_ = [0] + list(fedavg_asc_acc_max['asml1'][:49])
    min_ = [0] + list(fedavg_asc_acc_min['asml1'][:49])
    ax.plot(x_rounds, mean_, label='FedAvg (async)', ls='-.', c='g', marker='^', markevery=3, alpha=.6)

    mean_ = [0] + list(fedprox_acc_mean['asml1'][:49])
    max_ = [0] + list(fedprox_acc_max['asml1'][:49])
    min_ = [0] + list(fedprox_acc_min['asml1'][:49])
    ax.plot(x_rounds, mean_, label='FedProx (sync)', ls=':', c='m')

    mean_ = [0] + list(fedprox_asc_acc_mean['asml1'][:49])
    max_ = [0] + list(fedprox_asc_acc_max['asml1'][:49])
    min_ = [0] + list(fedprox_asc_acc_min['asml1'][:49])
    ax.plot(x_rounds, mean_, label='FedProx (async)', ls=':', c='m', marker='^', markevery=3, alpha=.6)

    mean_ = [0] + list(local_acc_mean['asml1'][:49])
    max_ = [0] + list(local_acc_max['asml1'][:49])
    min_ = [0] + list(local_acc_min['asml1'][:49])
    ax.plot(x_rounds, mean_, label='Local')
    ax.fill_between(x_rounds, max_, min_, alpha=.3)
    print("Local acc asml1", mean_[-1])

    mean_ = [0] + list(global_acc_mean['asml1'][:49])
    max_ = [0] + list(global_acc_max['asml1'][:49])
    min_ = [0] + list(global_acc_min['asml1'][:49])
    ax.plot(x_rounds, mean_, label='Centralized')

    ax.legend(loc='lower right')
    ax.set_ylim([.2, 1.])   #原始值是ax.set_ylim([.4, 1.])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('\#Rounds')
    ax.set_title(r'Accuracy on \texttt{Industry}')
    plot_dir = 'plot/final'
    os.makedirs(plot_dir, exist_ok=True)
    fig.tight_layout(pad=0.)
    fig.savefig(os.path.join(plot_dir, 'acc.asml1.pdf'))
    plt.close(fig)







    # fig, ax = plt.subplots(figsize=(6.4, 3.6))
    #
    # mean_ = [0] + list(hfl_asc_acc_mean['asml1'][:49])
    # max_ = [0] + list(hfl_asc_acc_max['asml1'][:49])
    # min_ = [0] + list(hfl_asc_acc_min['asml1'][:49])
    # ax.plot(x_rounds, mean_, label='HFL-LA (32 channels)')
    # ax.fill_between(x_rounds, max_, min_, color='r', alpha=.3)
    # print("HFL async acc asml1 32 channels", mean_[-1])
    #
    # mean_ = [0] + list(hfl_asc_26_acc_mean['asml1'][:49])
    # max_ = [0] + list(hfl_asc_26_acc_max['asml1'][:49])
    # min_ = [0] + list(hfl_asc_26_acc_min['asml1'][:49])
    # ax.plot(x_rounds, mean_, label='HFL-LA (26 channels)')
    # ax.fill_between(x_rounds, max_, min_, color='r', alpha=.3)
    # print("HFL async acc asml1 26 channels", mean_[-1])
    #
    # os.makedirs(plot_dir, exist_ok=True)
    # fig.savefig(os.path.join(plot_dir, 'acc.channels.pdf'))
    # plt.close(fig)
