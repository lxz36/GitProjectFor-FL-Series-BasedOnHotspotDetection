import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer


class FedMD():
    # parties, 里面存的模型。N_alignment,=5000需要对齐的公共数据集数量。N_rounds, =13总的循环轮数。N_logits_matching_round=1，每一轮后都进行逻辑匹配。N_private_training_round=10，私有数据训练10轮。
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data, private_test_data_asml1, N_alignment,
                 N_rounds,
                 N_logits_matching_round, logits_matching_batchsize,
                 N_private_training_round, private_training_batchsize):

        # 初始化参数
        self.N_parties = len(parties)  # 模型数量
        self.public_dataset = public_dataset  # 公共数据集
        self.private_data = private_data  # 私有数据集
        self.private_test_data = private_test_data  # 私有测试数据集
        self.private_test_data_asml1 = private_test_data_asml1  # 第二组私有测试数据集
        self.N_alignment = N_alignment  # 需要对齐的公共数据集数量

        self.N_rounds = N_rounds  # 总的训练轮数
        self.N_logits_matching_round = N_logits_matching_round  # 每一轮逻辑匹配的轮数
        self.logits_matching_batchsize = logits_matching_batchsize  # 逻辑匹配的批处理大小
        self.N_private_training_round = N_private_training_round  # 私有数据训练的轮数
        self.private_training_batchsize = private_training_batchsize  # 私有数据训练的批处理大小

        self.collaborative_parties = []  # 存储训练好的模型及其权重
        self.init_result = []  # 存储初始化结果

        print("start model initialization: ")
        for i in range(self.N_parties):  # 初始化每个模型
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])  # 克隆模型
            model_A_twin.set_weights(parties[i].get_weights())  # 设置权重
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])

            print("start full stack training ... ")

            model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                             batch_size=32, epochs=25, shuffle=True, verbose=0,
                             validation_data=(private_test_data["X"], private_test_data["y"]),
                             callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5)])

            print("full stack training done")

            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")  # 去掉模型的最后一层

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})
            # 记录初始化结果
            self.init_result.append({"val_acc": model_A_twin.history.history['val_accuracy'],
                                     "train_acc": model_A_twin.history.history['accuracy'],
                                     "val_loss": model_A_twin.history.history['val_loss'],
                                     "train_loss": model_A_twin.history.history['loss']})

            del model_A, model_A_twin
        # END FOR LOOP

        self.upper_bounds = []  # 初始化上界
        self.pooled_train_result = []  # 初始化联合训练结果

    def collaborative_training(self):
        # 开始协同训练
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        while True:
            # 每轮开始时生成新的对齐数据集
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)

            print("update logits ... ")
            # 更新logits
            logits = 0
            for d in self.collaborative_parties:
                d["model_logits"].set_weights(d["model_weights"])
                logits += d["model_logits"].predict(alignment_data["X"], verbose=0)

            logits /= self.N_parties

            # 测试性能
            print("test performance ... ")
            TPR_sum, FPR_sum, acc_sum = [], [], []
            for index, d in enumerate(self.collaborative_parties):
                if index < self.N_parties // 2:
                    y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
                    collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                    TPR = sum(self.private_test_data["y"] + y_pred == 2) / sum(self.private_test_data["y"])
                    FPR = sum(self.private_test_data["y"] - y_pred == -1) / (
                                len(self.private_test_data["y"]) - sum(self.private_test_data["y"]))
                    acc_sum.append(collaboration_performance[index][-1])
                    print("模型", index, "的TPR性能：", TPR)
                    TPR_sum.append(TPR)
                    print("模型", index, "的FPR性能：", FPR)
                    FPR_sum.append(FPR)
                    del y_pred
                else:
                    y_pred = d["model_classifier"].predict(self.private_test_data_asml1["X"], verbose=0).argmax(axis=1)
                    collaboration_performance[index].append(np.mean(self.private_test_data_asml1["y"] == y_pred))
                    TPR = sum(self.private_test_data_asml1["y"] + y_pred == 2) / sum(self.private_test_data_asml1["y"])
                    FPR = sum(self.private_test_data_asml1["y"] - y_pred == -1) / (
                                len(self.private_test_data_asml1["y"]) - sum(self.private_test_data_asml1["y"]))
                    acc_sum.append(collaboration_performance[index][-1])
                    print("模型", index, "的TPR性能：", TPR)
                    TPR_sum.append(TPR)
                    print("模型", index, "的FPR性能：", FPR)
                    FPR_sum.append(FPR)
                    del y_pred
            print("模型总的acc性能：", collaboration_performance)
            print("模型总的acc性能：", acc_sum)
            print("模型总的TPR性能：", TPR_sum)
            print("模型总的FPR性能：", FPR_sum)
            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = None
                weights_to_use = d["model_weights"]

                d["model_logits"].set_weights(weights_to_use)
                d["model_logits"].fit(alignment_data["X"], logits,
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)
                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                #lxztode 这里做卷积层的聚合

                # lxztode 私有数据训练，冻结卷积层，训练其他层
                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance
