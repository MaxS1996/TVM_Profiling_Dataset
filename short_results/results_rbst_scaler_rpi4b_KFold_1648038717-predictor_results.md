|                                   | device   | workload       | metric   | predictor   |          r2 |    max_error |         mae |         mape |            medae |    fit_time |   score_time |   dataset_min |   dataset_max |   dataset_mean |   dataset_median |   dataset_size |
|:----------------------------------|:---------|:---------------|:---------|:------------|------------:|-------------:|------------:|-------------:|-----------------:|------------:|-------------:|--------------:|--------------:|---------------:|-----------------:|---------------:|
| rasp4b-conv2d-time-xgb            | rasp4b   | conv2d         | time     | xgb         |  0.854299   |  9.49624     | 0.177556    |   11.1344    |      0.0256784   |  0.293129   |  0.00835907  |   5.57071e-06 |  40.071       |    0.62977     |      0.14018     |           1909 |
| rasp4b-conv2d-power-xgb           | rasp4b   | conv2d         | power    | xgb         | -0.439216   |  3.14242     | 0.402789    |    0.0928803 |      0.178966    |  0.307451   |  0.00416791  |   2.8355      |   7.6719      |    4.19455     |      4.066       |           1909 |
| rasp4b-conv2d-ws_size-xgb         | rasp4b   | conv2d         | ws_size  | xgb         |  0.849682   |  2.64115e+08 | 5.24428e+06 |    6.24304   | 964896           |  0.283496   |  0.00535071  | 288           |   9.2971e+08  |    2.64679e+07 |      8.11019e+06 |           1909 |
| rasp4b-conv2d-io_size-xgb         | rasp4b   | conv2d         | io_size  | xgb         |  0.91071    |  2.10738e+08 | 2.20478e+06 |   25.3843    | 226805           |  0.291919   |  0.00435883  |  16           |   8.07469e+08 |    1.4319e+07  |      1.8816e+06  |           1909 |
| rasp4b-conv2d-time-ert            | rasp4b   | conv2d         | time     | ert         |  0.74201    | 14.5905      | 0.221525    |    0.506081  |      0.0135674   |  0.55536    |  0.0239842   |   5.57071e-06 |  40.071       |    0.62977     |      0.14018     |           1909 |
| rasp4b-conv2d-power-ert           | rasp4b   | conv2d         | power    | ert         | -0.381115   |  3.10903     | 0.412849    |    0.0942312 |      0.187933    |  0.590573   |  0.0249361   |   2.8355      |   7.6719      |    4.19455     |      4.066       |           1909 |
| rasp4b-conv2d-ws_size-ert         | rasp4b   | conv2d         | ws_size  | ert         |  0.802327   |  2.98525e+08 | 6.4448e+06  |    0.286226  | 254064           |  0.554186   |  0.0243797   | 288           |   9.2971e+08  |    2.64679e+07 |      8.11019e+06 |           1909 |
| rasp4b-conv2d-io_size-ert         | rasp4b   | conv2d         | io_size  | ert         |  0.87322    |  2.77076e+08 | 2.5434e+06  |    0.146012  |  18553           |  0.474366   |  0.0211323   |  16           |   8.07469e+08 |    1.4319e+07  |      1.8816e+06  |           1909 |
| rasp4b-conv2d-time-dTr            | rasp4b   | conv2d         | time     | dTr         |  0.645606   | 12.8206      | 0.262074    |    0.554465  |      0.0159332   |  0.354065   |  0.00130183  |   5.57071e-06 |  40.071       |    0.62977     |      0.14018     |           1909 |
| rasp4b-conv2d-power-dTr           | rasp4b   | conv2d         | power    | dTr         | -0.756316   |  3.36783     | 0.447967    |    0.10215   |      0.186391    |  0.423274   |  0.0014658   |   2.8355      |   7.6719      |    4.19455     |      4.066       |           1909 |
| rasp4b-conv2d-ws_size-dTr         | rasp4b   | conv2d         | ws_size  | dTr         |  0.391731   |  5.92085e+08 | 1.00537e+07 |    0.402609  | 542594           |  0.32326    |  0.00138962  | 288           |   9.2971e+08  |    2.64679e+07 |      8.11019e+06 |           1909 |
| rasp4b-conv2d-io_size-dTr         | rasp4b   | conv2d         | io_size  | dTr         |  0.759999   |  2.3052e+08  | 4.22868e+06 |    0.168388  |     34           |  0.390017   |  0.00130367  |  16           |   8.07469e+08 |    1.4319e+07  |      1.8816e+06  |           1909 |
| rasp4b-conv2d-time-liR            | rasp4b   | conv2d         | time     | liR         |  0.180006   | 29.9534      | 0.842826    |  528.122     |      0.408413    |  0.00562596 |  0.000816703 |   5.57071e-06 |  40.071       |    0.62977     |      0.14018     |           1909 |
| rasp4b-conv2d-power-liR           | rasp4b   | conv2d         | power    | liR         |  0.00185025 |  3.19524     | 0.381536    |    0.0850151 |      0.242966    |  0.00650442 |  0.000951111 |   2.8355      |   7.6719      |    4.19455     |      4.066       |           1909 |
| rasp4b-conv2d-ws_size-liR         | rasp4b   | conv2d         | ws_size  | liR         |  0.287905   |  6.33371e+08 | 2.54854e+07 |  812.518     |      1.40111e+07 |  0.0053755  |  0.000876129 | 288           |   9.2971e+08  |    2.64679e+07 |      8.11019e+06 |           1909 |
| rasp4b-conv2d-io_size-liR         | rasp4b   | conv2d         | io_size  | liR         |  0.400908   |  3.40732e+08 | 1.53048e+07 | 2561.46      |      8.64009e+06 |  0.00584698 |  0.000912368 |  16           |   8.07469e+08 |    1.4319e+07  |      1.8816e+06  |           1909 |
| rasp4b-conv2d-time-kNN            | rasp4b   | conv2d         | time     | kNN         |  0.592763   | 20.7109      | 0.331644    |    1.63407   |      0.0192817   |  0.00531906 |  0.00916052  |   5.57071e-06 |  40.071       |    0.62977     |      0.14018     |           1909 |
| rasp4b-conv2d-power-kNN           | rasp4b   | conv2d         | power    | kNN         | -0.213155   |  3.20127     | 0.367498    |    0.0840318 |      0.181004    |  0.00565332 |  0.00999826  |   2.8355      |   7.6719      |    4.19455     |      4.066       |           1909 |
| rasp4b-conv2d-ws_size-kNN         | rasp4b   | conv2d         | ws_size  | kNN         |  0.735689   |  4.5887e+08  | 1.07286e+07 |    1.13662   | 866060           |  0.00586945 |  0.00885528  | 288           |   9.2971e+08  |    2.64679e+07 |      8.11019e+06 |           1909 |
| rasp4b-conv2d-io_size-kNN         | rasp4b   | conv2d         | io_size  | kNN         |  0.67121    |  3.12203e+08 | 5.84311e+06 |    1.57887   | 154081           |  0.00565356 |  0.0086416   |  16           |   8.07469e+08 |    1.4319e+07  |      1.8816e+06  |           1909 |
| rasp4b-conv2d-time-SVR            | rasp4b   | conv2d         | time     | SVR         |  0.330758   | 18.5151      | 0.310151    |   64.1515    |      0.0598719   |  0.0534706  |  0.0111874   |   5.57071e-06 |  40.071       |    0.62977     |      0.14018     |           1909 |
| rasp4b-conv2d-power-SVR           | rasp4b   | conv2d         | power    | SVR         | -0.0683181  |  3.08679     | 0.327761    |    0.0715749 |      0.164305    |  0.0813491  |  0.0220513   |   2.8355      |   7.6719      |    4.19455     |      4.066       |           1909 |
| rasp4b-conv2d-ws_size-SVR         | rasp4b   | conv2d         | ws_size  | SVR         | -0.0844027  |  7.51689e+08 | 2.27424e+07 |  185.805     |      7.29432e+06 |  0.0951064  |  0.0311331   | 288           |   9.2971e+08  |    2.64679e+07 |      8.11019e+06 |           1909 |
| rasp4b-conv2d-io_size-SVR         | rasp4b   | conv2d         | io_size  | SVR         | -0.0677441  |  6.55236e+08 | 1.41335e+07 |  765.034     |      1.83182e+06 |  0.0963113  |  0.0310647   |  16           |   8.07469e+08 |    1.4319e+07  |      1.8816e+06  |           1909 |
| rasp4b-conv2d-time-MLP            | rasp4b   | conv2d         | time     | MLP         |  0.777182   | 14.4476      | 0.353495    |   78.2788    |      0.113726    |  2.47775    |  0.00295562  |   5.57071e-06 |  40.071       |    0.62977     |      0.14018     |           1909 |
| rasp4b-conv2d-power-MLP           | rasp4b   | conv2d         | power    | MLP         | -0.11069    |  3.13265     | 0.384678    |    0.0871001 |      0.222399    |  4.24697    |  0.00336283  |   2.8355      |   7.6719      |    4.19455     |      4.066       |           1909 |
| rasp4b-conv2d-ws_size-MLP         | rasp4b   | conv2d         | ws_size  | MLP         |  0.413148   |  6.30488e+08 | 2.13606e+07 |   31.63      |      7.12202e+06 | 23.2332     |  0.00369596  | 288           |   9.2971e+08  |    2.64679e+07 |      8.11019e+06 |           1909 |
| rasp4b-conv2d-io_size-MLP         | rasp4b   | conv2d         | io_size  | MLP         |  0.597012   |  2.37844e+08 | 9.67404e+06 |  766.049     |      2.17016e+06 | 23.3533     |  0.00388592  |  16           |   8.07469e+08 |    1.4319e+07  |      1.8816e+06  |           1909 |
| rasp4b-dilated_conv2d-time-xgb    | rasp4b   | dilated_conv2d | time     | xgb         |  0.63393    | 15.9626      | 0.260014    |   13.253     |      0.034337    |  0.334462   |  0.0132563   |   5.57071e-06 |  40.071       |    0.620458    |      0.140151    |           2816 |
| rasp4b-dilated_conv2d-power-xgb   | rasp4b   | dilated_conv2d | power    | xgb         | -0.0401041  |  3.19692     | 0.483519    |    0.107323  |      0.250903    |  0.315585   |  0.00443155  |   2.8355      |   7.6719      |    4.32204     |      4.066       |           2816 |
| rasp4b-dilated_conv2d-ws_size-xgb | rasp4b   | dilated_conv2d | ws_size  | xgb         |  0.843061   |  3.99015e+08 | 5.9064e+06  |    7.77164   | 999842           |  0.375228   |  0.00438899  | 288           |   9.2971e+08  |    2.81409e+07 |      8.26658e+06 |           2816 |
| rasp4b-dilated_conv2d-io_size-xgb | rasp4b   | dilated_conv2d | io_size  | xgb         |  0.827389   |  4.61162e+08 | 3.39853e+06 |   69.2751    | 284950           |  0.323272   |  0.00434113  |  16           |   1.07053e+09 |    1.70016e+07 |      2.15827e+06 |           2816 |
| rasp4b-dilated_conv2d-time-ert    | rasp4b   | dilated_conv2d | time     | ert         |  0.794561   | 14.0586      | 0.215361    |    0.768101  |      0.0113462   |  0.910309   |  0.0304353   |   5.57071e-06 |  40.071       |    0.620458    |      0.140151    |           2816 |
| rasp4b-dilated_conv2d-power-ert   | rasp4b   | dilated_conv2d | power    | ert         | -0.238033   |  3.10745     | 0.493082    |    0.108849  |      0.209496    |  0.917242   |  0.0328285   |   2.8355      |   7.6719      |    4.32204     |      4.066       |           2816 |
| rasp4b-dilated_conv2d-ws_size-ert | rasp4b   | dilated_conv2d | ws_size  | ert         |  0.810882   |  4.32548e+08 | 6.00673e+06 |    0.246549  | 183109           |  0.85577    |  0.0293422   | 288           |   9.2971e+08  |    2.81409e+07 |      8.26658e+06 |           2816 |
| rasp4b-dilated_conv2d-io_size-ert | rasp4b   | dilated_conv2d | io_size  | ert         |  0.791816   |  5.80626e+08 | 3.20894e+06 |    0.118394  |  10676.9         |  0.745386   |  0.0258229   |  16           |   1.07053e+09 |    1.70016e+07 |      2.15827e+06 |           2816 |
| rasp4b-dilated_conv2d-time-dTr    | rasp4b   | dilated_conv2d | time     | dTr         | -0.382625   | 22.7951      | 0.342533    |    0.48948   |      0.011087    |  1.0057     |  0.00141251  |   5.57071e-06 |  40.071       |    0.620458    |      0.140151    |           2816 |
| rasp4b-dilated_conv2d-power-dTr   | rasp4b   | dilated_conv2d | power    | dTr         | -0.560416   |  3.44139     | 0.529455    |    0.115903  |      0.170531    |  1.33744    |  0.00140941  |   2.8355      |   7.6719      |    4.32204     |      4.066       |           2816 |
| rasp4b-dilated_conv2d-ws_size-dTr | rasp4b   | dilated_conv2d | ws_size  | dTr         |  0.565555   |  5.51794e+08 | 8.44524e+06 |    0.316813  | 216064           |  0.903303   |  0.00143278  | 288           |   9.2971e+08  |    2.81409e+07 |      8.26658e+06 |           2816 |
| rasp4b-dilated_conv2d-io_size-dTr | rasp4b   | dilated_conv2d | io_size  | dTr         |  0.748065   |  4.35441e+08 | 5.01029e+06 |    0.171385  |     -0           |  1.13921    |  0.00144649  |  16           |   1.07053e+09 |    1.70016e+07 |      2.15827e+06 |           2816 |
| rasp4b-dilated_conv2d-time-liR    | rasp4b   | dilated_conv2d | time     | liR         |  0.195502   | 23.3715      | 0.798634    |  502.038     |      0.453507    |  0.00840342 |  0.0012328   |   5.57071e-06 |  40.071       |    0.620458    |      0.140151    |           2816 |
| rasp4b-dilated_conv2d-power-liR   | rasp4b   | dilated_conv2d | power    | liR         |  0.0386531  |  3.28306     | 0.530294    |    0.114351  |      0.364447    |  0.00818831 |  0.00113147  |   2.8355      |   7.6719      |    4.32204     |      4.066       |           2816 |
| rasp4b-dilated_conv2d-ws_size-liR | rasp4b   | dilated_conv2d | ws_size  | liR         |  0.335584   |  7.24987e+08 | 2.53373e+07 |  426.06      |      1.28904e+07 |  0.00805479 |  0.00112087  | 288           |   9.2971e+08  |    2.81409e+07 |      8.26658e+06 |           2816 |
| rasp4b-dilated_conv2d-io_size-liR | rasp4b   | dilated_conv2d | io_size  | liR         |  0.336676   |  6.32648e+08 | 2.04993e+07 | 6995.78      |      1.20481e+07 |  0.0080353  |  0.0011363   |  16           |   1.07053e+09 |    1.70016e+07 |      2.15827e+06 |           2816 |
| rasp4b-dilated_conv2d-time-kNN    | rasp4b   | dilated_conv2d | time     | kNN         |  0.544251   | 13.1473      | 0.281602    |    3.09201   |      0.0299888   |  0.00658143 |  0.0217412   |   5.57071e-06 |  40.071       |    0.620458    |      0.140151    |           2816 |
| rasp4b-dilated_conv2d-power-kNN   | rasp4b   | dilated_conv2d | power    | kNN         | -0.15859    |  3.22728     | 0.495168    |    0.108458  |      0.215306    |  0.00675839 |  0.0185621   |   2.8355      |   7.6719      |    4.32204     |      4.066       |           2816 |
| rasp4b-dilated_conv2d-ws_size-kNN | rasp4b   | dilated_conv2d | ws_size  | kNN         |  0.677402   |  5.76966e+08 | 1.21475e+07 |    1.11249   |      1.20741e+06 |  0.00674206 |  0.0188829   | 288           |   9.2971e+08  |    2.81409e+07 |      8.26658e+06 |           2816 |
| rasp4b-dilated_conv2d-io_size-kNN | rasp4b   | dilated_conv2d | io_size  | kNN         |  0.621632   |  4.11995e+08 | 8.07326e+06 |    2.41016   | 310005           |  0.00682676 |  0.0187674   |  16           |   1.07053e+09 |    1.70016e+07 |      2.15827e+06 |           2816 |
| rasp4b-dilated_conv2d-time-SVR    | rasp4b   | dilated_conv2d | time     | SVR         |  0.267496   | 24.6073      | 0.401326    |   72.7377    |      0.0607874   |  0.101324   |  0.0229834   |   5.57071e-06 |  40.071       |    0.620458    |      0.140151    |           2816 |
| rasp4b-dilated_conv2d-power-SVR   | rasp4b   | dilated_conv2d | power    | SVR         | -0.0520289  |  3.43703     | 0.453158    |    0.0920296 |      0.201989    |  0.173843   |  0.0489179   |   2.8355      |   7.6719      |    4.32204     |      4.066       |           2816 |
| rasp4b-dilated_conv2d-ws_size-SVR | rasp4b   | dilated_conv2d | ws_size  | SVR         | -0.0982497  |  8.24118e+08 | 2.65167e+07 |  149.817     |      7.36633e+06 |  0.200003   |  0.0659881   | 288           |   9.2971e+08  |    2.81409e+07 |      8.26658e+06 |           2816 |
| rasp4b-dilated_conv2d-io_size-SVR | rasp4b   | dilated_conv2d | io_size  | SVR         | -0.0857349  |  6.81893e+08 | 1.61562e+07 |  401.11      |      2.05584e+06 |  0.206587   |  0.0679809   |  16           |   1.07053e+09 |    1.70016e+07 |      2.15827e+06 |           2816 |
| rasp4b-dilated_conv2d-time-MLP    | rasp4b   | dilated_conv2d | time     | MLP         |  0.649102   | 19.4482      | 0.305977    |   60.4502    |      0.0641002   |  3.1485     |  0.00346047  |   5.57071e-06 |  40.071       |    0.620458    |      0.140151    |           2816 |
| rasp4b-dilated_conv2d-power-MLP   | rasp4b   | dilated_conv2d | power    | MLP         | -0.0626212  |  3.37375     | 0.52288     |    0.112363  |      0.302038    |  7.78231    |  0.00454998  |   2.8355      |   7.6719      |    4.32204     |      4.066       |           2816 |
| rasp4b-dilated_conv2d-ws_size-MLP | rasp4b   | dilated_conv2d | ws_size  | MLP         |  0.424728   |  5.71766e+08 | 1.86352e+07 |   74.7626    |      5.86488e+06 | 36.5651     |  0.00529027  | 288           |   9.2971e+08  |    2.81409e+07 |      8.26658e+06 |           2816 |
| rasp4b-dilated_conv2d-io_size-MLP | rasp4b   | dilated_conv2d | io_size  | MLP         |  0.562169   |  4.54548e+08 | 1.27488e+07 |  772.736     |      2.93224e+06 | 38.6809     |  0.00601566  |  16           |   1.07053e+09 |    1.70016e+07 |      2.15827e+06 |           2816 |