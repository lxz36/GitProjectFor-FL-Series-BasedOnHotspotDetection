import matplotlib.pyplot as plt

# 2024-07-23的实验结果
acc_iccad_FedMD_async1 =  [0.0, 0.7216252157428629, 0.8323529411764706, 0.832219958690547, 0.951856095973743, 0.9515816427581134, 0.752559205500382, 0.7440766205472087, 0.8013199219081573, 0.8878886908298671, 0.8912613530260589, 0.806999971705854, 0.8054607701666525, 0.8088419206066266, 0.8625343066519537, 0.9500735647794472, 0.9473559120617944, 0.9459213988625752, 0.9490620490620489, 0.9436493789434965, 0.9426123984947514, 0.9223028605381547, 0.9355728149845797, 0.9355049090343208, 0.9318493053787172, 0.9342373312961548, 0.7081812522988994, 0.7956200662083015, 0.9092889681124975, 0.7361556743909685, 0.8586353733412556, 0.947602071131483, 0.9460359901536372, 0.9429222193928076, 0.9468579350932294, 0.9439945675239793, 0.8531052825170473, 0.9403474521121581, 0.6467928585575644, 0.8066293183940243, 0.8942577030812323, 0.7780692074809721, 0.8195540842599666, 0.7780932575050221, 0.874211300681889, 0.8645191409897292, 0.7962764903941375, 0.8747757688934159, 0.9438064114534702, 0.8376764847353083, 0.896456158220864]

acc_industry_FedMD_async1 = [0.0, 0.7409407805112589, 0.755958009455886, 0.8166519753185352, 0.8017068675374629, 0.8258354034778428, 0.7694606939658627, 0.8327670486417181, 0.8230146646365896, 0.8447551887170446, 0.8405160669925476, 0.8304511579453482, 0.8396586264925074, 0.8521836685631861, 0.8578812404840133, 0.858850869460694, 0.853193364852953, 0.8605256831476881, 0.867481368699415, 0.8636829874188636, 0.8641718086385127, 0.8567994230306916, 0.8626091834281594, 0.8659908646526164, 0.8672569917461335, 0.8622726179982371, 0.8413174132542671, 0.8442743809600127, 0.8665598204984374, 0.8623687795496435, 0.8556855517269011, 0.8621764564468307, 0.8703822421668402, 0.8748938216203221, 0.8715762480968026, 0.8642038624889814, 0.8643240644282393, 0.8666880359003125, 0.8596281753345622, 0.8645724817693725, 0.8712236557416461, 0.8634425835403479, 0.8637230547319497, 0.8614953121243689, 0.8693645324144562, 0.8562545075727221, 0.8653177337927719, 0.8706226460453562, 0.8673291129096883, 0.8571119480727623, 0.8645724817693725]

# 异步率为： 0.6 时 第 50 轮的测试结果： 2024-08-05的实验结果
acc_iccad_FedMD_async1 = [0.39323204029086384, 0.7229578700166935, 0.792552980788275, 0.9390162125456245, 0.9473163002574768, 0.9549599637834932, 0.7388973771326712, 0.9449693008516539, 0.9500636618283677, 0.9480448745154628, 0.8130025747672807, 0.8702953908836262, 0.9453484424072659, 0.9464094728800612, 0.9442096030331324, 0.9494652406417112, 0.8902653990889284, 0.9332385479444303, 0.9466669496081261, 0.9428246045893104, 0.7904875081345668, 0.8694027105791811, 0.8631539484480661, 0.8876269699799113, 0.7824052853464618, 0.814911014911015, 0.8894194041252865, 0.8882607588489941, 0.8901649548708372, 0.8929943694649577, 0.9421243244772658, 0.9406572730102141, 0.8951899951899952, 0.9242311065840477, 0.8521701609936905, 0.7915075120957474, 0.9453060011883542, 0.9420649067707891, 0.9421752539399598, 0.8460359901536373, 0.8898027898027898, 0.8873977873977875, 0.7511232775938659, 0.8412047647341765, 0.6812494694847636, 0.6807529072234954, 0.9426774750304162, 0.9339925869337634, 0.9384276943100472, 0.9428670458082223, 0.9416235180941064]
acc_industry_FedMD_async1 = [0.5761278948633705, 0.7361968106418784, 0.765205545316131, 0.7629617757833159, 0.8402275823383285, 0.8425194326468468, 0.7639634586104656, 0.8425114191842296, 0.8513262280631461, 0.8575927558297941, 0.8174292811924031, 0.8456446830675535, 0.8451077810722012, 0.8629136950076127, 0.8613350428720251, 0.8628896546197611, 0.8587547079092875, 0.8640836605497235, 0.8637551085824183, 0.8687555092555493, 0.8590912733392099, 0.8608221812645244, 0.8582658866896387, 0.8583700617036621, 0.8542190880679541, 0.8481609103293535, 0.8695408285920345, 0.8686272938536742, 0.8719849346902796, 0.8686914015546119, 0.8574164596522158, 0.8643000240403879, 0.8567112749419025, 0.8502123567593557, 0.8412212517028607, 0.8291129096882763, 0.8703742287042232, 0.8738520714800865, 0.8713678980687556, 0.8631460854235116, 0.859419825306515, 0.8677858802788684, 0.8496353874509175, 0.8513342415257632, 0.8466944466704062, 0.8683388091994552, 0.867761839891017, 0.8683067553489863, 0.8691722093116436, 0.8707749018350828, 0.8708069556855517]
#--------------------------------------------------------

# 异步率为： 0.6 时 第 50 轮的测试结果： 2024-08-06的实验结果，更换了网络架构，
# 两个数据集对应的模型分别是：CNN_16_16_16_32_32_32_FC240_320_240_2_asml1_acc0_59 和CNN_16_16_32_32_32_FC240_240_2_iccad1_acc0_07
acc_iccad_FedMD_async1_5Conv2FC = [0.07621735562912034, 0.5901041224570636, 0.7191763574116516, 0.9563987211046034, 0.9484480660951249, 0.9481340010751775, 0.9568372803666921, 0.9597232832526951, 0.9505828594063889, 0.8767209914268737, 0.8702911467617349, 0.8239509945392298, 0.8916475681181563, 0.912657386186798, 0.9120179384885267, 0.9011303511303511, 0.9253699459581813, 0.9526808703279291, 0.9140947288006112, 0.9353464618170501, 0.9520951815069463, 0.9160456101632573, 0.9067566420507596, 0.9022677757971875, 0.9242466683643153, 0.7618580765639589, 0.8596638655462184, 0.953922983334748, 0.8141315111903348, 0.8398834281187222, 0.9546897546897547, 0.9547039017627252, 0.949813258636788, 0.8694196870667458, 0.7718246894717483, 0.7681832328891153, 0.7710352827999888, 0.95528393175452, 0.9506253006253006, 0.9532920238802591, 0.9541648982825454, 0.8999717058540588, 0.8721132897603485, 0.9123093681917211, 0.8093667770138359, 0.8086452762923351, 0.8779418838242368, 0.9499476558300088, 0.9509563421328128, 0.8377302436125966, 0.8705316470022352]
acc_industry_FedMD_async1_5Conv2FC = [0.598124849747576, 0.7396105457168043, 0.7488260277265806, 0.6746373908165719, 0.7268450997676096, 0.7288644923471432, 0.7311002484173412, 0.7411170766888373, 0.7488100008013463, 0.7401474477121563, 0.7346261719689078, 0.74639794855357, 0.7384726340251622, 0.7499719528808397, 0.7439378155300906, 0.7430242807917301, 0.7426476480487219, 0.7438576809039186, 0.7420306114271977, 0.7533776744931485, 0.7526484493949835, 0.7514223896145524, 0.737807516627935, 0.7501642759836525, 0.7502524240724417, 0.729914255949996, 0.7265005208750701, 0.7481048160910329, 0.7317894062024202, 0.6984293613270294, 0.750140235595801, 0.749282795095761, 0.747495792932126, 0.7492267008574405, 0.7376071800625049, 0.7411811843897748, 0.7463418543152496, 0.7502524240724417, 0.7531052167641639, 0.7545636669604936, 0.7506611106659188, 0.7530330956006089, 0.7295215962817534, 0.7499799663434571, 0.7417501402355958, 0.7459011138713039, 0.7515746454042793, 0.7511899991986538, 0.7518390896706466, 0.7332879237118359, 0.7379998397307477]

#--------------------------------------------------------
acc_iccad_FedAvg_async1 =[0.0, 0.9822029822029823, 0.9825920267096737, 0.9834620716973659, 0.9851314263078969, 0.9348102877514644, 0.9798333474804062, 0.9797838327250092, 0.9678295560648502, 0.9731488555017966, 0.9626658744305804, 0.9540361599185129, 0.8782007752595987, 0.9533783210253798, 0.9505276858218036, 0.9480095068330362, 0.9448971507795039, 0.9399810429222194, 0.8606937724584783, 0.9393019834196303, 0.8100118835412953, 0.9324052853464618, 0.9222972017089666, 0.9437441643323996, 0.8898438163144047, 0.9320303879127408, 0.9255156608097785, 0.9462977110035935, 0.8641951730187024, 0.8960685284214696, 0.9225659960954079, 0.8563718416659594, 0.8947528506352036, 0.9042809042809041, 0.9126630450159862, 0.9352912882324647, 0.83043318337436, 0.9102014543191015, 0.9179611238434768, 0.8963090286619698, 0.8913434060492884, 0.8712970036499448, 0.919163625045978, 0.9156834450952098, 0.88292589763178, 0.8827207650737062, 0.9045496986673458, 0.8819851172792349, 0.894738703562233, 0.9130520895226779]



acc_industry_FedAvg_async1 =[0.0, 0.31256510938376475, 0.31256510938376475, 0.31256510938376475, 0.31256510938376475, 0.35607821139514384, 0.3149691481689238, 0.3126853113230227, 0.31276544594919464, 0.3129657825146246, 0.3136068595240003, 0.31532975398669766, 0.4055613430563346, 0.3172930523279109, 0.3146886769773219, 0.3137270614632583, 0.31412773459411814, 0.327349947912493, 0.44518791569837324, 0.31873547559900633, 0.5130619440660309, 0.33111627534257554, 0.3662152416058979, 0.3162112348745893, 0.4489141758153698, 0.34962737398830035, 0.37202500200336563, 0.31857520634666237, 0.5118599246734514, 0.4500761278948634, 0.33933007452520236, 0.5174292811924033, 0.4042791890375831, 0.3760718006250501, 0.35479605737639236, 0.3164516387531052, 0.5369821299783637, 0.40520073723856076, 0.345260036861928, 0.40291690039265965, 0.4267970189919065, 0.4574485135026845, 0.32378395704784035, 0.3338007853193365, 0.43793573202981007, 0.4326067793893741, 0.34013142078692205, 0.4473114832919304, 0.3768330795736838, 0.3386489302027406]


#------------------------------------------------------------------------以下是60%异步率的传统联邦蒸馏学习的实验结果
acc_iccad_TraditionalFedMD_async1 = [0.0, 0.6992586933763405, 0.8192966075319017, 0.9407322524969585, 0.9438573409161645, 0.9406035141329259, 0.9493902611549669, 0.9460260872025579, 0.9522408963585434, 0.9491214667685256, 0.9491893727187846, 0.9489177489177489, 0.9395424836601307, 0.9485060690943043, 0.9477987154457743, 0.9446707976119741, 0.9435404464816228, 0.9447358741476389, 0.9455182072829131, 0.9436465495289024, 0.9442746795687972, 0.9411453470277001, 0.9402625696743344, 0.9409288968112497, 0.940655858302917, 0.9396570749511927, 0.9455323543558837, 0.9425204425204425, 0.9393076422488187, 0.9397391279744222, 0.9377047788812494, 0.9275471804883569, 0.9326358826358827, 0.9388959624253742, 0.9354553942789238, 0.9412585236114648, 0.9391293891293891, 0.9331282007752597, 0.9365093512152336, 0.9393260334436805, 0.9401253430665195, 0.9374897433720962, 0.9365871601165718, 0.9372789519848344, 0.9353549500608324, 0.9305930452989276, 0.9361797244150185, 0.9371912401324167, 0.9359972271736978, 0.9341552782729254, 0.9360127889539654]
acc_industry_TraditionalFedMD_async1 = [0.0, 0.721243689398189, 0.8160269252343937, 0.794502764644603, 0.7931805433127656, 0.779565670326148, 0.7839009536020514, 0.7787242567513422, 0.7870983251863131, 0.8084542030611427, 0.8007131981729305, 0.8084862569116116, 0.8218607260197132, 0.816764163795176, 0.8160189117717765, 0.8148088789165799, 0.8216123086785801, 0.8106418783556375, 0.8251462456927638, 0.8237038224216684, 0.8345219969548842, 0.8265165478003045, 0.829753986697652, 0.8235675935571761, 0.8265726420386249, 0.830603413735075, 0.8377754627774661, 0.8258113630899911, 0.8307877233752705, 0.837767449314849, 0.8419825306514946, 0.8385607821139516, 0.834225498838048, 0.8377754627774662, 0.8309319657023799, 0.829064828912573, 0.8261960092956165, 0.8309800464780832, 0.8408125651093838, 0.8341373507492588, 0.8364612549082459, 0.8364131741325427, 0.8313727061463257, 0.8505970029649811, 0.8390496033336005, 0.8486096642359163, 0.8396666399551245, 0.8503886529369341, 0.8453642118759515, 0.8381601089830916, 0.8370702780671528]


#------------------------------------------------------------------------以下是60%异步率的联邦FedProx的实验结果

acc_iccad_FedProx_async1 = [0.0, 0.5494057724957556, 0.7136955291454443, 0.7457272212790039, 0.7504244482173175, 0.8555178268251273, 0.8960950764006792, 0.9101301641199774, 0.911941143180532, 0.9239954725523486, 0.9149405772495756, 0.9071873231465762, 0.9082625919637805, 0.911262026032824, 0.9069609507640068, 0.897962648556876, 0.8903791737408037, 0.886870401810979, 0.8873797396717601, 0.88907753254103, 0.8933786078098471, 0.8878324844368988, 0.8813242784380305, 0.8844934917940013, 0.8879456706281834, 0.8874363327674024, 0.8850594227504244, 0.8686474250141483, 0.8714204867006226, 0.8675721561969439, 0.8633276740237692, 0.8640633842671195, 0.8610639501980757, 0.8709111488398416, 0.8744199207696661, 0.8625919637804188, 0.8595359366157329, 0.8591397849462366, 0.8615166949632144, 0.8603282399547256, 0.856819468024901, 0.8567628749292586, 0.8515563101301641, 0.84538766270515, 0.8586870401810979, 0.8459535936615733, 0.8456140350877194, 0.8469722693831352, 0.8419354838709678, 0.8390492359932088]
acc_industry_FedProx_async1 = [0.5643286573146292, 0.6542685370741482, 0.6939478957915831, 0.7317835671342685, 0.7617635270541083, 0.7631262525050101, 0.7663326653306612, 0.7650501002004008, 0.7704208416833668, 0.7756312625250501, 0.7761122244488978, 0.7851703406813628, 0.7761923847695391, 0.7833266533066132, 0.7833266533066132, 0.7790781563126252, 0.779318637274549, 0.776993987975952, 0.776753507014028, 0.7858917835671343, 0.7837274549098197, 0.7856513026052104, 0.7774749498997996, 0.78436873747495, 0.777314629258517, 0.7796392785571142, 0.7867735470941885, 0.7830060120240481, 0.7814829659318637, 0.7868537074148297, 0.7800400801603207, 0.7827655310621242, 0.78436873747495, 0.7705811623246493, 0.7706613226452906, 0.7729859719438877, 0.7763527054108217, 0.775310621242485, 0.7790781563126252, 0.7874148296593186, 0.7857314629258517, 0.7798797595190381, 0.7745891783567134, 0.7791583166332665, 0.7708216432865732, 0.7632865731462924, 0.7672144288577154, 0.7674549098196394, 0.7762725450901804, 0.7733066132264528]

#------------------------------------------------------------------------上面是数据，下面是数据处理逻辑
len=min(len(acc_iccad_FedMD_async1),len(acc_iccad_FedAvg_async1))

rounds = list(range(1, len+1))

plt.figure(figsize=(14, 6))

# Plotting ICCAD results
plt.subplot(1, 2, 1)
plt.plot(rounds, acc_iccad_FedMD_async1[:len], label='FedMD-Conv ICCAD', marker='o')
plt.plot(rounds, acc_iccad_FedMD_async1_5Conv2FC[:len], label='FedMD-Conv-5Conv2FC ICCAD', marker='o')
plt.plot(rounds, acc_iccad_FedAvg_async1[:len], label='FedAvg ICCAD', marker='o')
plt.plot(rounds, acc_iccad_TraditionalFedMD_async1[:len], label='FedMD-Conv ICCAD', marker='o')
plt.plot(rounds, acc_iccad_FedProx_async1[:len], label='FedProx ICCAD', marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('ICCAD Test Results (Asynchronous Rate: 0.6)')
plt.legend()
plt.grid(True)

# Plotting Industry results
plt.subplot(1, 2, 2)
plt.plot(rounds, acc_industry_FedMD_async1[:len], label='FedMD-Conv Industry', marker='o')
plt.plot(rounds, acc_industry_FedMD_async1_5Conv2FC[:len], label='FedMD-Conv-5Conv2FC Industry', marker='o')
plt.plot(rounds, acc_industry_FedAvg_async1[:len], label='FedAvg Industry', marker='o')
plt.plot(rounds, acc_industry_TraditionalFedMD_async1[:len], label='Trad FedMD-Conv Industry', marker='o')
plt.plot(rounds, acc_industry_FedProx_async1[:len], label='FedProx Industry', marker='o')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Industry Test Results (Asynchronous Rate: 0.6)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
