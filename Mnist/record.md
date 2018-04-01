# 第一次
参数：
* batch_size:100
* steps : 20000
* drop out rate : 0.4
```
xtracting ./data_set/train-images-idx3-ubyte.gz
Extracting ./data_set/train-labels-idx1-ubyte.gz
Extracting ./data_set/t10k-images-idx3-ubyte.gz
Extracting ./data_set/t10k-labels-idx1-ubyte.gz
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_model_dir': './model', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x11a60df60>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0,'_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2018-04-01 18:16:56.951914: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:Restoring parameters from ./model/model.ckpt-1000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1001 into ./model/model.ckpt.
INFO:tensorflow:loss = 1.8940253, step = 1001
INFO:tensorflow:global_step/sec: 4.93021
INFO:tensorflow:loss = 1.9108245, step = 1101 (20.282 sec)
INFO:tensorflow:global_step/sec: 5.31525
INFO:tensorflow:loss = 1.6509628, step = 1201 (18.814 sec)
INFO:tensorflow:global_step/sec: 5.33765
INFO:tensorflow:loss = 1.4830036, step = 1301 (18.735 sec)
INFO:tensorflow:global_step/sec: 5.27063
INFO:tensorflow:loss = 1.274156, step = 1401 (18.973 sec)
INFO:tensorflow:global_step/sec: 5.28093
INFO:tensorflow:loss = 1.2035277, step = 1501 (18.936 sec)
INFO:tensorflow:global_step/sec: 5.30494
INFO:tensorflow:loss = 1.0988306, step = 1601 (18.850 sec)
INFO:tensorflow:global_step/sec: 5.26027
INFO:tensorflow:loss = 0.88351357, step = 1701 (19.011 sec)
INFO:tensorflow:global_step/sec: 5.30316
INFO:tensorflow:loss = 0.8530154, step = 1801 (18.857 sec)
INFO:tensorflow:global_step/sec: 5.31402
INFO:tensorflow:loss = 0.79744476, step = 1901 (18.818 sec)
INFO:tensorflow:global_step/sec: 5.31107
INFO:tensorflow:loss = 0.85303986, step = 2001 (18.829 sec)
INFO:tensorflow:global_step/sec: 5.28362
INFO:tensorflow:loss = 0.67792344, step = 2101 (18.926 sec)
INFO:tensorflow:global_step/sec: 5.29331
INFO:tensorflow:loss = 0.5468517, step = 2201 (18.892 sec)
INFO:tensorflow:global_step/sec: 5.31866
INFO:tensorflow:loss = 0.5705115, step = 2301 (18.802 sec)
INFO:tensorflow:global_step/sec: 5.31248
INFO:tensorflow:loss = 0.50173706, step = 2401 (18.824 sec)
INFO:tensorflow:global_step/sec: 5.2959
INFO:tensorflow:loss = 0.5019731, step = 2501 (18.883 sec)
^[[3~INFO:tensorflow:global_step/sec: 5.32137
INFO:tensorflow:loss = 0.48253393, step = 2601 (18.792 sec)
INFO:tensorflow:global_step/sec: 5.35122
INFO:tensorflow:loss = 0.675601, step = 2701 (18.687 sec)
INFO:tensorflow:global_step/sec: 5.426
INFO:tensorflow:loss = 0.59866524, step = 2801 (18.430 sec)
INFO:tensorflow:global_step/sec: 5.30598
INFO:tensorflow:loss = 0.67790306, step = 2901 (18.847 sec)
INFO:tensorflow:global_step/sec: 5.29604
INFO:tensorflow:loss = 0.4389447, step = 3001 (18.882 sec)
INFO:tensorflow:global_step/sec: 5.29163
INFO:tensorflow:loss = 0.3987281, step = 3101 (18.898 sec)
INFO:tensorflow:global_step/sec: 5.27341
INFO:tensorflow:loss = 0.43353993, step = 3201 (18.963 sec)
INFO:tensorflow:global_step/sec: 5.26928
INFO:tensorflow:loss = 0.47292686, step = 3301 (18.978 sec)
INFO:tensorflow:global_step/sec: 5.3218
INFO:tensorflow:loss = 0.3858462, step = 3401 (18.791 sec)
INFO:tensorflow:global_step/sec: 5.22531
INFO:tensorflow:loss = 0.4879969, step = 3501 (19.138 sec)
INFO:tensorflow:global_step/sec: 5.28844
INFO:tensorflow:loss = 0.33402267, step = 3601 (18.909 sec)
INFO:tensorflow:global_step/sec: 5.30872
INFO:tensorflow:loss = 0.3945431, step = 3701 (18.837 sec)
INFO:tensorflow:global_step/sec: 5.30959
INFO:tensorflow:loss = 0.42522022, step = 3801 (18.834 sec)
INFO:tensorflow:global_step/sec: 5.2861
INFO:tensorflow:loss = 0.41308543, step = 3901 (18.917 sec)
INFO:tensorflow:global_step/sec: 5.29713
INFO:tensorflow:loss = 0.5232542, step = 4001 (18.878 sec)
INFO:tensorflow:global_step/sec: 5.27534
INFO:tensorflow:loss = 0.32457536, step = 4101 (18.956 sec)
INFO:tensorflow:Saving checkpoints for 4175 into ./model/model.ckpt.
INFO:tensorflow:global_step/sec: 5.27619
INFO:tensorflow:loss = 0.34278065, step = 4201 (18.953 sec)
INFO:tensorflow:global_step/sec: 5.28861
INFO:tensorflow:loss = 0.32544762, step = 4301 (18.909 sec)
INFO:tensorflow:global_step/sec: 5.2565
INFO:tensorflow:loss = 0.23178826, step = 4401 (19.024 sec)
INFO:tensorflow:global_step/sec: 5.28612
INFO:tensorflow:loss = 0.37940794, step = 4501 (18.918 sec)
INFO:tensorflow:global_step/sec: 5.24888
INFO:tensorflow:loss = 0.27456352, step = 4601 (19.052 sec)
INFO:tensorflow:global_step/sec: 5.29698
INFO:tensorflow:loss = 0.3947495, step = 4701 (18.879 sec)
INFO:tensorflow:global_step/sec: 5.30297
INFO:tensorflow:loss = 0.24478872, step = 4801 (18.857 sec)
INFO:tensorflow:global_step/sec: 5.27884
INFO:tensorflow:loss = 0.40920544, step = 4901 (18.943 sec)
INFO:tensorflow:global_step/sec: 5.30333
INFO:tensorflow:loss = 0.39667743, step = 5001 (18.856 sec)
INFO:tensorflow:global_step/sec: 5.29304
INFO:tensorflow:loss = 0.28214514, step = 5101 (18.893 sec)
INFO:tensorflow:global_step/sec: 5.28475
INFO:tensorflow:loss = 0.21505924, step = 5201 (18.923 sec)
INFO:tensorflow:global_step/sec: 5.28313
INFO:tensorflow:loss = 0.30425376, step = 5301 (18.928 sec)
INFO:tensorflow:global_step/sec: 5.27291
INFO:tensorflow:loss = 0.2758269, step = 5401 (18.965 sec)
INFO:tensorflow:global_step/sec: 5.29524
INFO:tensorflow:loss = 0.29798403, step = 5501 (18.885 sec)
INFO:tensorflow:global_step/sec: 5.31235
INFO:tensorflow:loss = 0.3145978, step = 5601 (18.824 sec)
INFO:tensorflow:global_step/sec: 5.25205
INFO:tensorflow:loss = 0.26281658, step = 5701 (19.040 sec)
INFO:tensorflow:global_step/sec: 5.3018
INFO:tensorflow:loss = 0.2570549, step = 5801 (18.862 sec)
INFO:tensorflow:global_step/sec: 5.33631
INFO:tensorflow:loss = 0.35897198, step = 5901 (18.740 sec)
INFO:tensorflow:global_step/sec: 5.3944
INFO:tensorflow:loss = 0.3444453, step = 6001 (18.538 sec)
INFO:tensorflow:global_step/sec: 5.4318
INFO:tensorflow:loss = 0.23698719, step = 6101 (18.410 sec)
INFO:tensorflow:global_step/sec: 5.40993
INFO:tensorflow:loss = 0.4507212, step = 6201 (18.485 sec)
INFO:tensorflow:global_step/sec: 5.42875
INFO:tensorflow:loss = 0.17393224, step = 6301 (18.420 sec)
INFO:tensorflow:global_step/sec: 5.44074
INFO:tensorflow:loss = 0.26839378, step = 6401 (18.380 sec)
INFO:tensorflow:global_step/sec: 5.41514
INFO:tensorflow:loss = 0.2231307, step = 6501 (18.467 sec)
INFO:tensorflow:global_step/sec: 5.42544
INFO:tensorflow:loss = 0.30471298, step = 6601 (18.432 sec)
INFO:tensorflow:global_step/sec: 5.40267
INFO:tensorflow:loss = 0.25829306, step = 6701 (18.509 sec)
INFO:tensorflow:global_step/sec: 5.42102
INFO:tensorflow:loss = 0.28032738, step = 6801 (18.447 sec)
INFO:tensorflow:global_step/sec: 5.43336
INFO:tensorflow:loss = 0.2572932, step = 6901 (18.405 sec)
INFO:tensorflow:global_step/sec: 5.41248
INFO:tensorflow:loss = 0.2015658, step = 7001 (18.476 sec)
INFO:tensorflow:global_step/sec: 5.44487
INFO:tensorflow:loss = 0.2688294, step = 7101 (18.366 sec)
INFO:tensorflow:global_step/sec: 5.41087
INFO:tensorflow:loss = 0.29393178, step = 7201 (18.481 sec)
INFO:tensorflow:global_step/sec: 5.41655
INFO:tensorflow:loss = 0.23463413, step = 7301 (18.462 sec)
INFO:tensorflow:Saving checkpoints for 7384 into ./model/model.ckpt.
INFO:tensorflow:global_step/sec: 5.4165
INFO:tensorflow:loss = 0.2144733, step = 7401 (18.462 sec)
INFO:tensorflow:global_step/sec: 5.41524
INFO:tensorflow:loss = 0.36123675, step = 7501 (18.466 sec)
INFO:tensorflow:global_step/sec: 5.44121
INFO:tensorflow:loss = 0.2157811, step = 7601 (18.378 sec)
INFO:tensorflow:global_step/sec: 5.44017
INFO:tensorflow:loss = 0.07566687, step = 7701 (18.382 sec)
INFO:tensorflow:global_step/sec: 5.42068
INFO:tensorflow:loss = 0.27528092, step = 7801 (18.448 sec)
INFO:tensorflow:global_step/sec: 5.42553
INFO:tensorflow:loss = 0.2904592, step = 7901 (18.431 sec)
INFO:tensorflow:global_step/sec: 5.42149
INFO:tensorflow:loss = 0.2303336, step = 8001 (18.445 sec)
INFO:tensorflow:global_step/sec: 5.41335
INFO:tensorflow:loss = 0.21069914, step = 8101 (18.473 sec)
INFO:tensorflow:global_step/sec: 5.41485
INFO:tensorflow:loss = 0.18885128, step = 8201 (18.468 sec)
INFO:tensorflow:global_step/sec: 5.42322
INFO:tensorflow:loss = 0.24468687, step = 8301 (18.439 sec)
INFO:tensorflow:global_step/sec: 5.46011
INFO:tensorflow:loss = 0.18833114, step = 8401 (18.315 sec)
INFO:tensorflow:global_step/sec: 5.43947
INFO:tensorflow:loss = 0.15929897, step = 8501 (18.384 sec)
INFO:tensorflow:global_step/sec: 5.41423
INFO:tensorflow:loss = 0.2437787, step = 8601 (18.470 sec)
INFO:tensorflow:global_step/sec: 5.44341
INFO:tensorflow:loss = 0.14021178, step = 8701 (18.371 sec)
INFO:tensorflow:global_step/sec: 5.42095
INFO:tensorflow:loss = 0.26429433, step = 8801 (18.447 sec)
INFO:tensorflow:global_step/sec: 5.41968
INFO:tensorflow:loss = 0.33238578, step = 8901 (18.451 sec)
INFO:tensorflow:global_step/sec: 5.42261
INFO:tensorflow:loss = 0.2953681, step = 9001 (18.441 sec)
INFO:tensorflow:global_step/sec: 5.42688
INFO:tensorflow:loss = 0.124791555, step = 9101 (18.427 sec)
INFO:tensorflow:global_step/sec: 5.43514
INFO:tensorflow:loss = 0.14218064, step = 9201 (18.399 sec)
INFO:tensorflow:global_step/sec: 5.39614
INFO:tensorflow:loss = 0.3824462, step = 9301 (18.532 sec)
INFO:tensorflow:global_step/sec: 5.4023
INFO:tensorflow:loss = 0.14318222, step = 9401 (18.511 sec)
INFO:tensorflow:global_step/sec: 5.42405
INFO:tensorflow:loss = 0.31773156, step = 9501 (18.436 sec)
INFO:tensorflow:global_step/sec: 5.41308
INFO:tensorflow:loss = 0.2869185, step = 9601 (18.474 sec)
INFO:tensorflow:global_step/sec: 5.41697
INFO:tensorflow:loss = 0.31568873, step = 9701 (18.461 sec)
INFO:tensorflow:global_step/sec: 5.41528
INFO:tensorflow:loss = 0.16015959, step = 9801 (18.466 sec)
INFO:tensorflow:global_step/sec: 5.37687
INFO:tensorflow:loss = 0.14972313, step = 9901 (18.598 sec)
INFO:tensorflow:global_step/sec: 5.43979
INFO:tensorflow:loss = 0.14033344, step = 10001 (18.383 sec)
INFO:tensorflow:global_step/sec: 5.39899
INFO:tensorflow:loss = 0.09052553, step = 10101 (18.522 sec)
INFO:tensorflow:global_step/sec: 5.44566
INFO:tensorflow:loss = 0.16095866, step = 10201 (18.363 sec)
INFO:tensorflow:global_step/sec: 5.43681
INFO:tensorflow:loss = 0.220125, step = 10301 (18.393 sec)
INFO:tensorflow:global_step/sec: 5.41728
INFO:tensorflow:loss = 0.27457574, step = 10401 (18.459 sec)
INFO:tensorflow:global_step/sec: 5.42004
INFO:tensorflow:loss = 0.21239391, step = 10501 (18.450 sec)
INFO:tensorflow:global_step/sec: 5.39739
INFO:tensorflow:loss = 0.28338653, step = 10601 (18.527 sec)
INFO:tensorflow:Saving checkpoints for 10637 into ./model/model.ckpt.
INFO:tensorflow:global_step/sec: 5.39869
INFO:tensorflow:loss = 0.29852906, step = 10701 (18.523 sec)
INFO:tensorflow:global_step/sec: 5.39731
INFO:tensorflow:loss = 0.1014, step = 10801 (18.528 sec)
INFO:tensorflow:global_step/sec: 5.41084
INFO:tensorflow:loss = 0.10772569, step = 10901 (18.481 sec)
INFO:tensorflow:global_step/sec: 5.43391
INFO:tensorflow:loss = 0.18982233, step = 11001 (18.403 sec)
INFO:tensorflow:global_step/sec: 5.43251
INFO:tensorflow:loss = 0.11876792, step = 11101 (18.408 sec)
INFO:tensorflow:global_step/sec: 5.41249
INFO:tensorflow:loss = 0.07511938, step = 11201 (18.476 sec)
INFO:tensorflow:global_step/sec: 5.43599
INFO:tensorflow:loss = 0.17992589, step = 11301 (18.396 sec)
INFO:tensorflow:global_step/sec: 5.41471
INFO:tensorflow:loss = 0.137232, step = 11401 (18.468 sec)
INFO:tensorflow:global_step/sec: 5.42113
INFO:tensorflow:loss = 0.12399651, step = 11501 (18.446 sec)
INFO:tensorflow:global_step/sec: 5.41608
INFO:tensorflow:loss = 0.15999094, step = 11601 (18.463 sec)
INFO:tensorflow:global_step/sec: 5.43609
INFO:tensorflow:loss = 0.20587985, step = 11701 (18.396 sec)
INFO:tensorflow:global_step/sec: 5.4506
INFO:tensorflow:loss = 0.20865054, step = 11801 (18.347 sec)
INFO:tensorflow:global_step/sec: 5.40331
INFO:tensorflow:loss = 0.17315555, step = 11901 (18.507 sec)
INFO:tensorflow:global_step/sec: 5.39004
INFO:tensorflow:loss = 0.089819826, step = 12001 (18.553 sec)
INFO:tensorflow:global_step/sec: 5.42138
INFO:tensorflow:loss = 0.15539733, step = 12101 (18.445 sec)
INFO:tensorflow:global_step/sec: 5.41974
INFO:tensorflow:loss = 0.29658473, step = 12201 (18.451 sec)
INFO:tensorflow:global_step/sec: 5.44249
INFO:tensorflow:loss = 0.19188419, step = 12301 (18.374 sec)
INFO:tensorflow:global_step/sec: 5.40156
INFO:tensorflow:loss = 0.18447137, step = 12401 (18.513 sec)
INFO:tensorflow:global_step/sec: 5.38444
INFO:tensorflow:loss = 0.13838434, step = 12501 (18.572 sec)
INFO:tensorflow:global_step/sec: 5.41764
INFO:tensorflow:loss = 0.17170055, step = 12601 (18.458 sec)
INFO:tensorflow:global_step/sec: 5.43516
INFO:tensorflow:loss = 0.19867876, step = 12701 (18.399 sec)
INFO:tensorflow:global_step/sec: 5.45751
INFO:tensorflow:loss = 0.10047167, step = 12801 (18.323 sec)
INFO:tensorflow:global_step/sec: 5.41762
INFO:tensorflow:loss = 0.1518425, step = 12901 (18.458 sec)
INFO:tensorflow:global_step/sec: 5.39144
INFO:tensorflow:loss = 0.1719103, step = 13001 (18.548 sec)
INFO:tensorflow:global_step/sec: 5.40158
INFO:tensorflow:loss = 0.20796782, step = 13101 (18.513 sec)
INFO:tensorflow:global_step/sec: 5.42897
INFO:tensorflow:loss = 0.18730852, step = 13201 (18.420 sec)
INFO:tensorflow:global_step/sec: 5.43543
INFO:tensorflow:loss = 0.25501335, step = 13301 (18.398 sec)
INFO:tensorflow:global_step/sec: 5.42931
INFO:tensorflow:loss = 0.14322807, step = 13401 (18.419 sec)
INFO:tensorflow:global_step/sec: 5.43105
INFO:tensorflow:loss = 0.10541705, step = 13501 (18.413 sec)
INFO:tensorflow:global_step/sec: 5.42792
INFO:tensorflow:loss = 0.16685387, step = 13601 (18.423 sec)
INFO:tensorflow:global_step/sec: 5.40277
INFO:tensorflow:loss = 0.19508025, step = 13701 (18.509 sec)
INFO:tensorflow:global_step/sec: 5.41065
INFO:tensorflow:loss = 0.1344194, step = 13801 (18.482 sec)
INFO:tensorflow:Saving checkpoints for 13889 into ./model/model.ckpt.
INFO:tensorflow:global_step/sec: 5.40305
INFO:tensorflow:loss = 0.22316104, step = 13901 (18.508 sec)
INFO:tensorflow:global_step/sec: 5.41678
INFO:tensorflow:loss = 0.07592347, step = 14001 (18.461 sec)
INFO:tensorflow:global_step/sec: 5.40095
INFO:tensorflow:loss = 0.0904867, step = 14101 (18.515 sec)
INFO:tensorflow:global_step/sec: 5.40599
INFO:tensorflow:loss = 0.1635569, step = 14201 (18.498 sec)
INFO:tensorflow:global_step/sec: 5.42713
INFO:tensorflow:loss = 0.18289424, step = 14301 (18.426 sec)
INFO:tensorflow:global_step/sec: 5.40133
INFO:tensorflow:loss = 0.14477529, step = 14401 (18.514 sec)
INFO:tensorflow:global_step/sec: 5.42781
INFO:tensorflow:loss = 0.23233469, step = 14501 (18.424 sec)
INFO:tensorflow:global_step/sec: 5.37441
INFO:tensorflow:loss = 0.12539582, step = 14601 (18.607 sec)
INFO:tensorflow:global_step/sec: 5.4195
INFO:tensorflow:loss = 0.10374195, step = 14701 (18.452 sec)
INFO:tensorflow:global_step/sec: 5.40851
INFO:tensorflow:loss = 0.074391104, step = 14801 (18.489 sec)
INFO:tensorflow:global_step/sec: 5.43183
INFO:tensorflow:loss = 0.18704265, step = 14901 (18.410 sec)
INFO:tensorflow:global_step/sec: 5.40119
INFO:tensorflow:loss = 0.10407868, step = 15001 (18.515 sec)
INFO:tensorflow:global_step/sec: 5.41475
INFO:tensorflow:loss = 0.2130631, step = 15101 (18.468 sec)
INFO:tensorflow:global_step/sec: 5.42401
INFO:tensorflow:loss = 0.080755875, step = 15201 (18.437 sec)
INFO:tensorflow:global_step/sec: 5.42453
INFO:tensorflow:loss = 0.09857542, step = 15301 (18.435 sec)
INFO:tensorflow:global_step/sec: 5.44771
INFO:tensorflow:loss = 0.109959304, step = 15401 (18.356 sec)
INFO:tensorflow:global_step/sec: 5.41094
INFO:tensorflow:loss = 0.22319032, step = 15501 (18.481 sec)
INFO:tensorflow:global_step/sec: 5.40917
INFO:tensorflow:loss = 0.14366728, step = 15601 (18.487 sec)
INFO:tensorflow:global_step/sec: 5.41788
INFO:tensorflow:loss = 0.07954, step = 15701 (18.457 sec)
INFO:tensorflow:global_step/sec: 5.40456
INFO:tensorflow:loss = 0.10888591, step = 15801 (18.503 sec)
INFO:tensorflow:global_step/sec: 5.41764
INFO:tensorflow:loss = 0.17059368, step = 15901 (18.458 sec)
INFO:tensorflow:global_step/sec: 5.44974
INFO:tensorflow:loss = 0.18021107, step = 16001 (18.349 sec)
INFO:tensorflow:global_step/sec: 5.37772
INFO:tensorflow:loss = 0.23797372, step = 16101 (18.595 sec)
INFO:tensorflow:global_step/sec: 5.44083
INFO:tensorflow:loss = 0.12808289, step = 16201 (18.380 sec)
INFO:tensorflow:global_step/sec: 5.45327
INFO:tensorflow:loss = 0.15642262, step = 16301 (18.338 sec)
INFO:tensorflow:global_step/sec: 5.4258
INFO:tensorflow:loss = 0.077790804, step = 16401 (18.430 sec)
INFO:tensorflow:global_step/sec: 5.44913
INFO:tensorflow:loss = 0.16967966, step = 16501 (18.351 sec)
INFO:tensorflow:global_step/sec: 5.40601
INFO:tensorflow:loss = 0.12181302, step = 16601 (18.498 sec)
INFO:tensorflow:global_step/sec: 5.42007
INFO:tensorflow:loss = 0.09681549, step = 16701 (18.450 sec)
INFO:tensorflow:global_step/sec: 5.41471
INFO:tensorflow:loss = 0.05651807, step = 16801 (18.468 sec)
INFO:tensorflow:global_step/sec: 5.39335
INFO:tensorflow:loss = 0.26329863, step = 16901 (18.541 sec)
INFO:tensorflow:global_step/sec: 5.42789
INFO:tensorflow:loss = 0.15903467, step = 17001 (18.423 sec)
INFO:tensorflow:global_step/sec: 5.44734
INFO:tensorflow:loss = 0.117520265, step = 17101 (18.358 sec)
INFO:tensorflow:Saving checkpoints for 17140 into ./model/model.ckpt.
INFO:tensorflow:global_step/sec: 5.41198
INFO:tensorflow:loss = 0.21721186, step = 17201 (18.477 sec)
INFO:tensorflow:global_step/sec: 5.44753
INFO:tensorflow:loss = 0.120911844, step = 17301 (18.357 sec)
INFO:tensorflow:global_step/sec: 5.398
INFO:tensorflow:loss = 0.08891951, step = 17401 (18.525 sec)
INFO:tensorflow:global_step/sec: 5.40029
INFO:tensorflow:loss = 0.1375764, step = 17501 (18.518 sec)
INFO:tensorflow:global_step/sec: 5.41372
INFO:tensorflow:loss = 0.14669476, step = 17601 (18.472 sec)
INFO:tensorflow:global_step/sec: 5.41352
INFO:tensorflow:loss = 0.15706015, step = 17701 (18.472 sec)
INFO:tensorflow:global_step/sec: 5.43837
INFO:tensorflow:loss = 0.07472824, step = 17801 (18.388 sec)
INFO:tensorflow:global_step/sec: 5.41527
INFO:tensorflow:loss = 0.06254514, step = 17901 (18.466 sec)
INFO:tensorflow:global_step/sec: 5.42253
INFO:tensorflow:loss = 0.1660394, step = 18001 (18.442 sec)
INFO:tensorflow:global_step/sec: 5.42081
INFO:tensorflow:loss = 0.21000499, step = 18101 (18.447 sec)
INFO:tensorflow:global_step/sec: 5.41276
INFO:tensorflow:loss = 0.19731332, step = 18201 (18.475 sec)
INFO:tensorflow:global_step/sec: 5.43568
INFO:tensorflow:loss = 0.23332176, step = 18301 (18.397 sec)
INFO:tensorflow:global_step/sec: 5.43522
INFO:tensorflow:loss = 0.19713002, step = 18401 (18.399 sec)
INFO:tensorflow:global_step/sec: 5.43307
INFO:tensorflow:loss = 0.20985964, step = 18501 (18.408 sec)
INFO:tensorflow:global_step/sec: 5.42528
INFO:tensorflow:loss = 0.07943372, step = 18601 (18.430 sec)
INFO:tensorflow:global_step/sec: 5.4045
INFO:tensorflow:loss = 0.2197078, step = 18701 (18.503 sec)
INFO:tensorflow:global_step/sec: 5.42519
INFO:tensorflow:loss = 0.14505495, step = 18801 (18.433 sec)
INFO:tensorflow:global_step/sec: 5.43586
INFO:tensorflow:loss = 0.17182122, step = 18901 (18.396 sec)
INFO:tensorflow:global_step/sec: 5.40945
INFO:tensorflow:loss = 0.15667784, step = 19001 (18.486 sec)
INFO:tensorflow:global_step/sec: 5.42029
INFO:tensorflow:loss = 0.24240123, step = 19101 (18.449 sec)
INFO:tensorflow:global_step/sec: 5.40823
INFO:tensorflow:loss = 0.16916768, step = 19201 (18.490 sec)
INFO:tensorflow:global_step/sec: 5.41453
INFO:tensorflow:loss = 0.08802043, step = 19301 (18.469 sec)
INFO:tensorflow:global_step/sec: 5.41224
INFO:tensorflow:loss = 0.11506888, step = 19401 (18.477 sec)
INFO:tensorflow:global_step/sec: 5.43699
INFO:tensorflow:loss = 0.15148267, step = 19501 (18.393 sec)
INFO:tensorflow:global_step/sec: 5.42248
INFO:tensorflow:loss = 0.08495232, step = 19601 (18.442 sec)
INFO:tensorflow:global_step/sec: 5.44001
INFO:tensorflow:loss = 0.09957839, step = 19701 (18.382 sec)
INFO:tensorflow:global_step/sec: 4.8414
INFO:tensorflow:loss = 0.08855563, step = 19801 (20.655 sec)
INFO:tensorflow:global_step/sec: 5.44244
INFO:tensorflow:loss = 0.08091113, step = 19901 (18.374 sec)
INFO:tensorflow:global_step/sec: 5.4379
INFO:tensorflow:loss = 0.09071449, step = 20001 (18.389 sec)
INFO:tensorflow:global_step/sec: 5.3841
INFO:tensorflow:loss = 0.11953393, step = 20101 (18.573 sec)
INFO:tensorflow:global_step/sec: 5.13893
INFO:tensorflow:loss = 0.06151581, step = 20201 (19.459 sec)
INFO:tensorflow:global_step/sec: 5.40808
INFO:tensorflow:loss = 0.10031681, step = 20301 (18.491 sec)
INFO:tensorflow:Saving checkpoints for 20376 into ./model/model.ckpt.
INFO:tensorflow:global_step/sec: 5.4186
INFO:tensorflow:loss = 0.11758187, step = 20401 (18.455 sec)
INFO:tensorflow:global_step/sec: 5.43334
INFO:tensorflow:loss = 0.27151766, step = 20501 (18.405 sec)
INFO:tensorflow:global_step/sec: 5.30495
INFO:tensorflow:loss = 0.11899259, step = 20601 (18.850 sec)
INFO:tensorflow:global_step/sec: 5.34925
INFO:tensorflow:loss = 0.096694686, step = 20701 (18.694 sec)
INFO:tensorflow:global_step/sec: 5.3283
INFO:tensorflow:loss = 0.08044182, step = 20801 (18.768 sec)
INFO:tensorflow:global_step/sec: 5.3439
INFO:tensorflow:loss = 0.08766919, step = 20901 (18.713 sec)
INFO:tensorflow:Saving checkpoints for 21000 into ./model/model.ckpt.
INFO:tensorflow:Loss for final step: 0.14325519.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2018-04-01-11:18:54
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./model/model.ckpt-21000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2018-04-01-11:19:01
INFO:tensorflow:Saving dict for global step 21000: accuracy = 0.9711, global_step = 21000, loss = 0.09560749
{'accuracy': 0.9711, 'loss': 0.09560749, 'global_step': 21000}
```

# 第二次
* batch_size : 200
* steps : 10000
* drop out rate : 0.4