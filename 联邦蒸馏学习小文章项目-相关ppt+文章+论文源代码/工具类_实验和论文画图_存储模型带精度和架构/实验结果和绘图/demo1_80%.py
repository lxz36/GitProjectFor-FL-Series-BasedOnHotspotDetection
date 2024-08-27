import matplotlib.pyplot as plt

# 2024-07-23的实验结果
acc_iccad_FedMD_async1 = [0.0, 0.8254279489573607, 0.8330235124352772, 0.9581416404945816, 0.9454771807712984, 0.9497396938573409, 0.944455762102821, 0.93877288289053, 0.9339430721783664, 0.8908369408369408, 0.9457785134255723, 0.9402215431627197, 0.9381320204849617, 0.9390926067396655, 0.942884022295787, 0.9150001414707297, 0.9400814871403107, 0.8795065500947855, 0.8770180799592564, 0.8944458591517416, 0.9424468777409954, 0.9445250827603768, 0.9438658291599469, 0.9450824774354185, 0.9336403248167955, 0.9115892821775174, 0.9389327448150979, 0.9406360524007583, 0.9343278725631666, 0.9375293551764141, 0.9430990578049402, 0.9411609088079675, 0.9408298673004556, 0.9349036584330703, 0.9418569447981213, 0.943210819681408, 0.9138400814871404, 0.9403686727216138, 0.8800384800384802, 0.9191721132897603, 0.9400164106046459, 0.9394349659055541, 0.9409161644455761, 0.936823416235181, 0.934453781512605, 0.8995189995189994, 0.9012859689330277, 0.9021998698469286, 0.8824887530769884, 0.8790764790764791, 0.9396938573409163]

acc_industry_FedMD_async1 = [0.0, 0.793036300985656, 0.8438176135908325, 0.8291289366135107, 0.8436733712637231, 0.8490183508293935, 0.858121644362529, 0.8649010337366777, 0.8724016347463739, 0.8536341052968988, 0.8616635948393301, 0.8666960493629297, 0.8744610946389935, 0.8732270213959452, 0.8731709271576248, 0.8750060100969629, 0.8738520714800865, 0.8690119400592996, 0.8730026444426636, 0.8727942944146164, 0.8764804872185271, 0.8721692443304752, 0.8795576568635306, 0.8702460133023481, 0.8813446590271656, 0.8737398830034457, 0.8768811603493869, 0.879317252985015, 0.8785399471111466, 0.8738280310922351, 0.8803349627373989, 0.8833239842936133, 0.8781072201298181, 0.8778187354755989, 0.8806555012420867, 0.8770574565269653, 0.8768330795736837, 0.8770654699895826, 0.8708470229986377, 0.8765445949194648, 0.8781553009055212, 0.876801025723215, 0.8739722734193446, 0.8806635147047039, 0.8785720009616155, 0.8770254026764965, 0.8736757753025082, 0.8802227742607581, 0.8765045276063788, 0.880911932045837, 0.8813927398028689]

# 异步率为： 0.8 时 第 50 轮的测试结果：MD卷积层聚合的结果
acc_iccad_FedMD_async1= [0.3978494623655914, 0.8410299943406905, 0.8385964912280702, 0.9655913978494624, 0.9674023769100171, 0.942161856253537, 0.9481041312959819, 0.9513865308432372, 0.9504810413129599, 0.9661573288058858, 0.9652518392756082, 0.9685342388228635, 0.9693265421618562, 0.9647425014148274, 0.9705715902659875, 0.96893039049236, 0.9669496321448783, 0.9425014148273909, 0.9669496321448783, 0.9658743633276738, 0.9689303904923598, 0.9625353706847765, 0.9671194114318054, 0.9666666666666666, 0.9667798528579514, 0.9638936049801924, 0.9668364459535936, 0.965025466893039, 0.9674589700056593, 0.96723259762309, 0.9646859083191851, 0.9655913978494624, 0.9618562535370685, 0.9684210526315791, 0.9564233163554047, 0.9572722127900397, 0.9647425014148274, 0.9642897566496889, 0.964516129032258, 0.96723259762309, 0.9671760045274477, 0.9658743633276741, 0.9438596491228071, 0.9679117147707981, 0.9673457838143747, 0.9658743633276741, 0.9673457838143747, 0.9649688737973967, 0.9673457838143745, 0.9661007357102435, 0.9633276740237691]
acc_industry_FedMD_async1= [0.5643286573146292, 0.758316633266533, 0.7771543086172346, 0.7744288577154308, 0.7656112224448898, 0.7855711422845693, 0.7985571142284569, 0.7898196392785571, 0.7986372745490982, 0.7996793587174349, 0.7939879759519037, 0.7954308617234468, 0.8139478957915832, 0.7991983967935872, 0.8023246492985973, 0.8096993987975951, 0.8067334669338677, 0.802565130260521, 0.7992785571142285, 0.8000801603206412, 0.8132264529058117, 0.8091382765531062, 0.8026452905811624, 0.8054509018036073, 0.814188376753507, 0.8155511022044089, 0.8097795591182365, 0.8123446893787575, 0.8134669338677355, 0.8024048096192384, 0.8041683366733465, 0.8117835671342686, 0.8112224448897795, 0.8134669338677355, 0.8069739478957916, 0.8117034068136272, 0.8128256513026052, 0.8159519038076152, 0.8179559118236472, 0.8185170340681364, 0.8128256513026052, 0.8088977955911825, 0.7967134268537074, 0.8096993987975953, 0.8067334669338677, 0.8125851703406812, 0.8121042084168337, 0.8113026052104209, 0.8118637274549098, 0.8129859719438878, 0.8123446893787575]



acc_iccad_FedAvg_async1 =[0.0, 0.98285374755963, 0.9838369691310869, 0.9712390006507654, 0.9735661941544296, 0.9661036131624365, 0.9469555498967264, 0.9260320289732056, 0.9493322581557877, 0.8908482584953173, 0.94011544011544, 0.922615510850805, 0.9232379820615113, 0.8498075998075997, 0.9217879070820247, 0.9158178422884304, 0.9414240443652208, 0.9061624649859944, 0.9118708089296325, 0.8562303709362533, 0.9157754010695187, 0.9228489375548199, 0.9115878674702204, 0.8729522111875052, 0.9154924596101066, 0.9282248252836489, 0.9174093879976233, 0.884715502362561, 0.9103287779758368, 0.915053900348018, 0.909020173726056, 0.912825736355148, 0.9074993633817163, 0.8414679002914298, 0.8903035961859491, 0.9037857567269331, 0.9280126191890897, 0.8938120702826586, 0.8971295588942649, 0.8873822256175197, 0.908708938120703, 0.9111988229635288, 0.9073366720425543, 0.8979076479076479, 0.8965848965848966, 0.8944699091757915, 0.8623984947514358, 0.8954743513567042, 0.8441487706193589, 0.9053419347536995]

acc_industry_FedAvg_async1 =    [0.0, 0.31256510938376475, 0.31256510938376475, 0.3277906883564388, 0.3128455805753666, 0.3142078692202901, 0.3152896866736117, 0.3277906883564388, 0.3131661190800545, 0.39983171728503886, 0.3210994470710794, 0.34269572882442506, 0.3459011138713038, 0.4774821700456767, 0.35070919144162194, 0.3855677538264284, 0.34501963298341215, 0.38973475438737076, 0.3688196169564869, 0.4873787963779149, 0.35788124048401315, 0.3622485776103854, 0.37855597403638114, 0.4551646766567834, 0.3871704463498678, 0.33708630499238723, 0.3549162593156503, 0.45772898469428636, 0.38821219649010336, 0.3964259956727302, 0.4007131981729305, 0.4082458530330956, 0.4119320458370062, 0.5776905200737239, 0.49058418142479365, 0.4394582899270775, 0.3314768811603494, 0.4320057696930844, 0.438256270534498, 0.47940540107380397, 0.4376953281512942, 0.3703421748537543, 0.4129337286641558, 0.42295055693565187, 0.4261559419825306, 0.4790848625691161, 0.5744050004006731, 0.4632182065870662, 0.6090632262200497, 0.4283997115153458]

acc_iccad_TraditionalFedMD_async1 = [0.0, 0.7947783153665506, 0.9376991200520612, 0.9624890360184477, 0.9592875534052006, 0.948944628356393, 0.9500933706816059, 0.9416702034349094, 0.9526271114506407, 0.9480137509549275, 0.9337535014005602, 0.9487706193588545, 0.9436918201624085, 0.9435715700421582, 0.9414084825849531, 0.943871487989135, 0.9415966386554622, 0.9389426477661772, 0.9375873581755935, 0.9407959143253262, 0.9425147836912544, 0.9324533853945619, 0.9390473361061596, 0.937015816427581, 0.9329386299974536, 0.9381801205330618, 0.9406233200350848, 0.9315210932857992, 0.9335441247205953, 0.9372110460345755, 0.9339954163483576, 0.9294542059247941, 0.9327674504145091, 0.9304529892765186, 0.9356548680078092, 0.9375095492742551, 0.9372945137651021, 0.93114053702289, 0.9288105141046318, 0.931796961208726, 0.9331168831168831, 0.9263064821888352, 0.930813739637269, 0.9316781257957729, 0.9261211555329203, 0.9184576861047449, 0.9299281328693093, 0.9279843250431487, 0.9198115609880315, 0.9308703279291516, 0.9289180318592084]

acc_industry_TraditionalFedMD_async1 = [0.0, 0.7659668242647648, 0.7723695808959051, 0.7439858963057937, 0.7526244090071319, 0.7653978684189437, 0.7775703181344658, 0.8038785159067233, 0.7906643160509657, 0.8111707668883724, 0.8132863210193124, 0.8131500921548203, 0.8108342014584501, 0.8050645083740683, 0.8197932526644763, 0.8167000560942383, 0.8197772257392419, 0.8216203221411972, 0.8352432085904319, 0.8287603173331195, 0.8342655661511339, 0.8352111547399632, 0.8361166760157064, 0.8292090712396826, 0.8249779629778027, 0.8229665838608863, 0.8338328391698052, 0.8401073803990705, 0.8345780911932046, 0.8350188316371504, 0.8392018591233272, 0.838256270534498, 0.8336485295296099, 0.840291690039266, 0.8359323663755109, 0.832438496674413, 0.8385046878756311, 0.8373427357961376, 0.8360125010016828, 0.842206907604776, 0.8393621283756711, 0.8342495392258995, 0.8353153297539867, 0.8416459652215723, 0.838624889814889, 0.8411651574645405, 0.8398269092074685, 0.8277746614312044, 0.8449475118198574, 0.8438576809039186, 0.8412372786280953]


acc_iccad_FedProx_async1 = [0.0, 0.7130164119977364, 0.7828522920203735, 0.8877758913412563, 0.9101867572156197, 0.9225806451612903, 0.9251273344651952, 0.9169779286926995, 0.9142614601018675, 0.906451612903226, 0.9061686474250141, 0.8946802490096208, 0.8910016977928692, 0.8890209394453876, 0.8925863044708546, 0.8920769666100735, 0.8772495755517827, 0.8746462931522354, 0.8730616864742501, 0.8760611205432938, 0.8667798528579512, 0.8642897566496888, 0.8628749292586304, 0.8700056593095642, 0.8692699490662139, 0.8649122807017544, 0.873740803621958, 0.8557441992076967, 0.8628749292586304, 0.8626485568760611, 0.854272778720996, 0.847934352009055, 0.858177702320317, 0.8698358800226371, 0.8637804187889078, 0.8638370118845501, 0.8385398981324277, 0.8308998302207131, 0.8358234295415959, 0.8515563101301641, 0.8410299943406905, 0.8289756649688738, 0.832258064516129, 0.8335597057159028, 0.8478211658177702, 0.8436898698358799, 0.8166949632144878, 0.8522920203735145, 0.8345783814374647, 0.8281267685342388]
acc_industry_FedProx_async1 = [0.0, 0.6549098196392785, 0.7003607214428857, 0.7442084168336673, 0.7543086172344688, 0.7674549098196393, 0.7745891783567134, 0.7678557114228457, 0.7695390781563127, 0.7793987975951904, 0.7821242484969939, 0.7802004008016032, 0.773066132264529, 0.7756312625250501, 0.780681362725451, 0.7717835671342685, 0.7802805611222445, 0.7741883767535069, 0.7740280561122244, 0.7765130260521043, 0.7812424849699399, 0.7735470941883767, 0.7692184368737476, 0.7674549098196393, 0.7745891783567135, 0.7711422845691384, 0.7745090180360721, 0.7714629258517034, 0.7708216432865732, 0.7661723446893788, 0.7623246492985973, 0.7710621242484971, 0.7546292585170341, 0.7567935871743487, 0.76, 0.7582364729458917, 0.751503006012024, 0.7523046092184369, 0.7607214428857716, 0.7498997995991985, 0.7582364729458918, 0.7620841683366735, 0.7486973947895791, 0.7606412825651303, 0.7634468937875751, 0.758557114228457, 0.7292184368737475, 0.7187975951903807, 0.7543887775551102, 0.729619238476954]



#------------------------------------------------------------------------上面是数据，下面是数据处理逻辑
len=min(len(acc_iccad_FedMD_async1),len(acc_iccad_FedAvg_async1))

rounds = list(range(1, len+1))

plt.figure(figsize=(14, 6))

# Plotting ICCAD results
plt.subplot(1, 2, 1)
plt.plot(rounds, acc_iccad_FedMD_async1[:len], label='FedMD-Conv ICCAD', marker='o')
plt.plot(rounds, acc_iccad_FedAvg_async1[:len], label='FedAvg ICCAD', marker='o')
plt.plot(rounds, acc_iccad_TraditionalFedMD_async1[:len], label='Trad FedMD-Conv ICCAD', marker='o')
plt.plot(rounds, acc_iccad_FedProx_async1[:len], label='FedProx ICCAD', marker='o')

plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('ICCAD Test Results (Asynchronous Rate: 0.8)')
plt.legend()
plt.grid(True)

# Plotting Industry results
plt.subplot(1, 2, 2)
plt.plot(rounds, acc_industry_FedMD_async1[:len], label='FedMD-Conv Industry', marker='o')
plt.plot(rounds, acc_industry_FedAvg_async1[:len], label='FedAvg Industry', marker='o')
plt.plot(rounds, acc_industry_TraditionalFedMD_async1[:len], label='Trad FedMD-Conv Industry', marker='o')
plt.plot(rounds, acc_industry_FedProx_async1[:len], label='FedProx Industry', marker='o')

plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Industry Test Results (Asynchronous Rate: 0.8)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

