import matplotlib.pyplot as plt

# FedMD第25轮的测试结果 (10节点, 异步率: 1)
acc_iccad_FedMD_async1 = [
    0.9273873185637891, 0.9302054154995332, 0.9304897716662422, 0.9362518744871686,
    0.9276844070961718, 0.9353323147440795, 0.9333262597968479, 0.932033217327335,
    0.9422912599383186, 0.9344933933169228, 0.9362306538777126, 0.9337959426194722,
    0.9298701298701297, 0.9320204849616613, 0.9289024700789407, 0.9360835243188184,
    0.9324420677361853, 0.9301035565741447, 0.9366720425543955, 0.9337464278640748,
    0.931000481000481, 0.92999037999038, 0.9305407011289365, 0.9299493534787653,
    0.9313923549217666
]
acc_industry_FedMD_async1 = [
    0.706106258514304, 0.8267489382162033, 0.849130539306034, 0.8622886449234715,
    0.8595881080214761, 0.8628095199935892, 0.8674252744610946, 0.866784197451719,
    0.8662312685311322, 0.8682907284237519, 0.8722173251061784, 0.8696530170686755,
    0.8654379357320299, 0.8704623767930123, 0.8718807596762561, 0.8674493148489463,
    0.8718727462136389, 0.8692443304751982, 0.8726099847744211, 0.8693244651013703,
    0.869108101610706, 0.8741966503726261, 0.8767930122605978, 0.8743729465502044,
    0.8719128135267249
]

# FedAvg第25轮的测试结果 (10节点, 异步率: 1)
acc_iccad_FedAvg_async1 = [
    0.9728234728234728, 0.9462481962481963, 0.9369960105254223, 0.904974110856464,
    0.8882664176781823, 0.8858967829556065, 0.8709645474351356, 0.8468154938743174,
    0.8783422459893048, 0.8628794952324365, 0.8528845881787058, 0.8481311716605834,
    0.8481948334889511, 0.8447075800016977, 0.8411708117590472, 0.8308575955634779,
    0.8470701411877883, 0.85474492827434, 0.8416447387035623, 0.8293862999745352,
    0.8199431287666583, 0.8369762046232635, 0.8408949438361203, 0.8321944939591999,
    0.836870101575984
]
acc_industry_FedAvg_async1 = [
    0.31380719608943025, 0.3572000961615514, 0.3702219729144964, 0.386609503966664,
    0.42363170125811356, 0.4133344017950156, 0.43320778908566393, 0.48721852712557095,
    0.4518390896706467, 0.4672650052087507, 0.4650212356759356, 0.47736196810641884,
    0.4846942864011539, 0.49727542271015307, 0.4838929401394342, 0.512941742126773,
    0.4973154900232391, 0.5009215482009777, 0.523239041589871, 0.5433127654459492,
    0.5551726901194006, 0.5345780911932045, 0.520514464300024, 0.5094158185752063,
    0.5352592355156663
]

# LocalTraining第25轮的训练结果
accLocal_iccad = [0.9106499165322696] * 25
accLocal_industry = [0.7899511178780351] * 25

rounds = list(range(1, 26))

plt.figure(figsize=(14, 6))

# Plotting ICCAD results
plt.subplot(1, 2, 1)
plt.plot(rounds, acc_iccad_FedMD_async1, label='FedMD ICCAD', marker='o')
plt.plot(rounds, acc_iccad_FedAvg_async1, label='FedAvg ICCAD', marker='o')
plt.plot(rounds, accLocal_iccad, label='LocalTraining ICCAD', linestyle='--', marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('ICCAD Test Results (Asynchronous Rate: 1)')
plt.legend()
plt.grid(True)

# Plotting Industry results
plt.subplot(1, 2, 2)
plt.plot(rounds, acc_industry_FedMD_async1, label='FedMD Industry', marker='o')
plt.plot(rounds, acc_industry_FedAvg_async1, label='FedAvg Industry', marker='o')
plt.plot(rounds, accLocal_industry, label='LocalTraining Industry', linestyle='--', marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Industry Test Results (Asynchronous Rate: 1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
