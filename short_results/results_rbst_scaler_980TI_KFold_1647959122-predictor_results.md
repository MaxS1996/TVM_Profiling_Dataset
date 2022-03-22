|                                        | device     | workload         | metric   | predictor   |        r2 |     max_error |          mae |        mape |        medae |   fit_time |   score_time |   dataset_min |   dataset_max |   dataset_mean |   dataset_median |   dataset_size |
|:---------------------------------------|:-----------|:-----------------|:---------|:------------|----------:|--------------:|-------------:|------------:|-------------:|-----------:|-------------:|--------------:|--------------:|---------------:|-----------------:|---------------:|
| cuda_980ti-avg_pool2d-time-xgb         | cuda_980ti | avg_pool2d       | time     | xgb         |  0.936727 |   10.376      |  0.599845    |  0.753438   |  0.291458    |   0.463253 |   0.0100695  |    0.00695231 |     37.963    |       3.31406  |        1.46691   |           4692 |
| cuda_980ti-avg_pool2d-power-xgb        | cuda_980ti | avg_pool2d       | power    | xgb         |  0.957179 |   55.4249     |  2.32841     |  0.014965   |  1.44611     |   0.402341 |   0.00491291 |   65.936      |    207.425    |     160.817    |      152.66      |           4692 |
| cuda_980ti-avg_pool2d-memory-xgb       | cuda_980ti | avg_pool2d       | memory   | xgb         |  0.966861 |    1.09065    |  0.114605    |  0.16589    |  0.0619752   |   0.386734 |   0.0056268  |    0.0974731  |      4.89948  |       0.972265 |        0.513855  |           4692 |
| cuda_980ti-avg_pool2d-time-ert         | cuda_980ti | avg_pool2d       | time     | ert         |  0.918191 |   11.2465     |  0.574848    |  0.313287   |  0.208934    |   1.53772  |   0.0453643  |    0.00695231 |     37.963    |       3.31406  |        1.46691   |           4692 |
| cuda_980ti-avg_pool2d-power-ert        | cuda_980ti | avg_pool2d       | power    | ert         |  0.929714 |   46.8751     |  2.75348     |  0.0181112  |  1.42928     |   1.26585  |   0.0464116  |   65.936      |    207.425    |     160.817    |      152.66      |           4692 |
| cuda_980ti-avg_pool2d-memory-ert       | cuda_980ti | avg_pool2d       | memory   | ert         |  0.960811 |    1.40166    |  0.105092    |  0.113096   |  0.0407493   |   1.60059  |   0.0452363  |    0.0974731  |      4.89948  |       0.972265 |        0.513855  |           4692 |
| cuda_980ti-avg_pool2d-time-dTr         | cuda_980ti | avg_pool2d       | time     | dTr         |  0.762486 |   13.6758     |  1.12597     |  0.486307   |  0.401153    |   1.63487  |   0.00148433 |    0.00695231 |     37.963    |       3.31406  |        1.46691   |           4692 |
| cuda_980ti-avg_pool2d-power-dTr        | cuda_980ti | avg_pool2d       | power    | dTr         |  0.84796  |   82.294      |  3.85199     |  0.0251368  |  1.8055      |   1.43954  |   0.00158232 |   65.936      |    207.425    |     160.817    |      152.66      |           4692 |
| cuda_980ti-avg_pool2d-memory-dTr       | cuda_980ti | avg_pool2d       | memory   | dTr         |  0.872411 |    2.00943    |  0.207161    |  0.206322   |  0.0806885   |   1.74391  |   0.00157166 |    0.0974731  |      4.89948  |       0.972265 |        0.513855  |           4692 |
| cuda_980ti-avg_pool2d-time-MLP         | cuda_980ti | avg_pool2d       | time     | MLP         |  0.979794 |    9.26329    |  0.275583    |  0.332844   |  0.12163     |   9.41329  |   0.0132698  |    0.00695231 |     37.963    |       3.31406  |        1.46691   |           4692 |
| cuda_980ti-avg_pool2d-power-MLP        | cuda_980ti | avg_pool2d       | power    | MLP         |  0.919147 |   68.8203     |  3.71579     |  0.0239979  |  2.58927     |  26.2715   |   0.00635439 |   65.936      |    207.425    |     160.817    |      152.66      |           4692 |
| cuda_980ti-avg_pool2d-memory-MLP       | cuda_980ti | avg_pool2d       | memory   | MLP         |  0.990968 |    0.913112   |  0.0536676   |  0.09291    |  0.0304201   |   5.97346  |   0.0077731  |    0.0974731  |      4.89948  |       0.972265 |        0.513855  |           4692 |
| cuda_980ti-conv2d-time-xgb             | cuda_980ti | conv2d           | time     | xgb         |  0.796181 | 1118.83       | 13.6207      | 11.0972     |  1.54397     |   0.35521  |   0.0056209  |    0.00618766 |   3395.59     |      28.5176   |        0.864422  |           2690 |
| cuda_980ti-conv2d-power-xgb            | cuda_980ti | conv2d           | power    | xgb         |  0.921368 |   64.641      |  8.87703     |  0.0534059  |  5.73321     |   0.353562 |   0.00574499 |   80.68       |    273.31     |     183.758    |      191.673     |           2690 |
| cuda_980ti-conv2d-memory-xgb           | cuda_980ti | conv2d           | memory   | xgb         |  0.948918 |    0.639257   |  0.0173729   |  0.0530427  |  0.00326733  |   0.361391 |   0.00668275 |    0.0979614  |      3.82013  |       0.208935 |        0.117737  |           2690 |
| cuda_980ti-conv2d-time-ert             | cuda_980ti | conv2d           | time     | ert         |  0.553187 |  716.394      | 13.4278      |  1.64902    |  0.406777    |   0.77381  |   0.0304459  |    0.00618766 |   3395.59     |      28.5176   |        0.864422  |           2690 |
| cuda_980ti-conv2d-power-ert            | cuda_980ti | conv2d           | power    | ert         |  0.865544 |   60.0852     | 12.2834      |  0.070516   |  8.58247     |   0.725692 |   0.0324776  |   80.68       |    273.31     |     183.758    |      191.673     |           2690 |
| cuda_980ti-conv2d-memory-ert           | cuda_980ti | conv2d           | memory   | ert         |  0.903693 |    1.55953    |  0.0246252   |  0.0404384  |  0.000926252 |   0.724183 |   0.0279188  |    0.0979614  |      3.82013  |       0.208935 |        0.117737  |           2690 |
| cuda_980ti-conv2d-time-dTr             | cuda_980ti | conv2d           | time     | dTr         |  0.351708 | 1798.45       | 24.0691      |  2.29963    |  0.405009    |   0.930451 |   0.00145757 |    0.00618766 |   3395.59     |      28.5176   |        0.864422  |           2690 |
| cuda_980ti-conv2d-power-dTr            | cuda_980ti | conv2d           | power    | dTr         |  0.853922 |   85.8204     | 10.4456      |  0.0626963  |  4.22625     |   0.767268 |   0.0014348  |   80.68       |    273.31     |     183.758    |      191.673     |           2690 |
| cuda_980ti-conv2d-memory-dTr           | cuda_980ti | conv2d           | memory   | dTr         |  0.738059 |    1.53574    |  0.0385831   |  0.0813288  |  0.00227356  |   0.763916 |   0.00142431 |    0.0979614  |      3.82013  |       0.208935 |        0.117737  |           2690 |
| cuda_980ti-conv2d-time-MLP             | cuda_980ti | conv2d           | time     | MLP         |  0.43354  | 1803.37       | 27.4851      | 39.8392     |  4.43099     |   6.48615  |   0.0043745  |    0.00618766 |   3395.59     |      28.5176   |        0.864422  |           2690 |
| cuda_980ti-conv2d-power-MLP            | cuda_980ti | conv2d           | power    | MLP         |  0.774209 |   77.5543     | 16.9205      |  0.0967253  | 13.2435      |  13.7931   |   0.00488287 |   80.68       |    273.31     |     183.758    |      191.673     |           2690 |
| cuda_980ti-conv2d-memory-MLP           | cuda_980ti | conv2d           | memory   | MLP         |  0.889984 |    1.17051    |  0.0410402   |  0.181818   |  0.0177586   |   4.00845  |   0.00438559 |    0.0979614  |      3.82013  |       0.208935 |        0.117737  |           2690 |
| cuda_980ti-dense-time-xgb              | cuda_980ti | dense            | time     | xgb         |  0.998256 |   20.5969     |  0.87484     |  0.384301   |  0.44552     |   0.966317 |   0.0136681  |    0.00684414 |    256.193    |      23.6654   |        8.68221   |          23629 |
| cuda_980ti-dense-power-xgb             | cuda_980ti | dense            | power    | xgb         |  0.926149 |   25.0202     |  2.28234     |  0.0129178  |  1.61781     |   0.954161 |   0.00893492 |   83.273      |    235.344    |     178.337    |      179.268     |          23629 |
| cuda_980ti-dense-memory-xgb            | cuda_980ti | dense            | memory   | xgb         |  0.999764 |    0.00798389 |  0.00118554  |  0.00599945 |  0.000878774 |   1.9295   |   0.0140507  |    0.0974731  |      0.573181 |       0.213768 |        0.182556  |          23629 |
| cuda_980ti-dense-time-ert              | cuda_980ti | dense            | time     | ert         |  0.998003 |   25.4427     |  0.611553    |  0.0356895  |  0.112769    |   3.38141  |   0.237702   |    0.00684414 |    256.193    |      23.6654   |        8.68221   |          23629 |
| cuda_980ti-dense-power-ert             | cuda_980ti | dense            | power    | ert         |  0.871659 |   32.6568     |  3.01624     |  0.0169076  |  2.12069     |   3.27416  |   0.238292   |   83.273      |    235.344    |     178.337    |      179.268     |          23629 |
| cuda_980ti-dense-memory-ert            | cuda_980ti | dense            | memory   | ert         |  0.999943 |    0.0070609  |  0.000507454 |  0.00235004 |  0.000334909 |   3.44462  |   0.221399   |    0.0974731  |      0.573181 |       0.213768 |        0.182556  |          23629 |
| cuda_980ti-dense-time-dTr              | cuda_980ti | dense            | time     | dTr         |  0.99637  |   29.6252     |  0.941748    |  0.0624384  |  0.239324    |  13.4079   |   0.00306469 |    0.00684414 |    256.193    |      23.6654   |        8.68221   |          23629 |
| cuda_980ti-dense-power-dTr             | cuda_980ti | dense            | power    | dTr         |  0.819264 |   42.2944     |  3.28455     |  0.0184723  |  1.80406     |  18.5397   |   0.00301683 |   83.273      |    235.344    |     178.337    |      179.268     |          23629 |
| cuda_980ti-dense-memory-dTr            | cuda_980ti | dense            | memory   | dTr         |  0.999705 |    0.0127258  |  0.00115335  |  0.00521572 |  0.000915527 |  12.6657   |   0.00278699 |    0.0974731  |      0.573181 |       0.213768 |        0.182556  |          23629 |
| cuda_980ti-dense-time-MLP              | cuda_980ti | dense            | time     | MLP         |  0.997502 |   31.0861     |  0.860547    |  0.239621   |  0.366819    |  19.0504   |   0.0731357  |    0.00684414 |    256.193    |      23.6654   |        8.68221   |          23629 |
| cuda_980ti-dense-power-MLP             | cuda_980ti | dense            | power    | MLP         |  0.764172 |   48.0025     |  3.9811      |  0.0224744  |  2.82049     |  75.4834   |   0.028644   |   83.273      |    235.344    |     178.337    |      179.268     |          23629 |
| cuda_980ti-dense-memory-MLP            | cuda_980ti | dense            | memory   | MLP         |  0.999959 |    0.00465579 |  0.000508744 |  0.00286777 |  0.000411853 |   8.71873  |   0.0314665  |    0.0974731  |      0.573181 |       0.213768 |        0.182556  |          23629 |
| cuda_980ti-depthwise_conv2d-time-xgb   | cuda_980ti | depthwise_conv2d | time     | xgb         |  0.616139 |  117.686      |  0.973887    |  3.04772    |  0.0762384   |   0.358125 |   0.00909281 |    0.00595117 |    260.158    |       1.69531  |        0.0716767 |           3083 |
| cuda_980ti-depthwise_conv2d-power-xgb  | cuda_980ti | depthwise_conv2d | power    | xgb         |  0.941776 |   46.2145     |  5.62912     |  0.0384598  |  3.84437     |   0.348076 |   0.00501311 |   81.001      |    244.784    |     150.137    |      156.638     |           3083 |
| cuda_980ti-depthwise_conv2d-memory-xgb | cuda_980ti | depthwise_conv2d | memory   | xgb         |  0.946158 |    0.655778   |  0.0155458   |  0.0519113  |  0.00269383  |   0.380205 |   0.00835776 |    0.0974731  |      3.81964  |       0.187668 |        0.106506  |           3083 |
| cuda_980ti-depthwise_conv2d-time-ert   | cuda_980ti | depthwise_conv2d | time     | ert         |  0.744883 |   62.8269     |  0.838802    |  1.09093    |  0.019618    |   0.994652 |   0.0338694  |    0.00595117 |    260.158    |       1.69531  |        0.0716767 |           3083 |
| cuda_980ti-depthwise_conv2d-power-ert  | cuda_980ti | depthwise_conv2d | power    | ert         |  0.924499 |   41.6483     |  6.1452      |  0.0417122  |  3.61122     |   0.872648 |   0.032772   |   81.001      |    244.784    |     150.137    |      156.638     |           3083 |
| cuda_980ti-depthwise_conv2d-memory-ert | cuda_980ti | depthwise_conv2d | memory   | ert         |  0.919246 |    1.14609    |  0.0150735   |  0.0276648  |  0.000377197 |   0.854208 |   0.029467   |    0.0974731  |      3.81964  |       0.187668 |        0.106506  |           3083 |
| cuda_980ti-depthwise_conv2d-time-dTr   | cuda_980ti | depthwise_conv2d | time     | dTr         |  0.421685 |  104.282      |  1.12026     |  1.08857    |  0.0155883   |   1.10906  |   0.00147855 |    0.00595117 |    260.158    |       1.69531  |        0.0716767 |           3083 |
| cuda_980ti-depthwise_conv2d-power-dTr  | cuda_980ti | depthwise_conv2d | power    | dTr         |  0.870339 |   68.44       |  7.30089     |  0.0500443  |  3.36362     |   1.16082  |   0.00142211 |   81.001      |    244.784    |     150.137    |      156.638     |           3083 |
| cuda_980ti-depthwise_conv2d-memory-dTr | cuda_980ti | depthwise_conv2d | memory   | dTr         |  0.768285 |    1.63321    |  0.0314018   |  0.0614347  |  0.00115967  |   1.00634  |   0.00145501 |    0.0974731  |      3.81964  |       0.187668 |        0.106506  |           3083 |
| cuda_980ti-depthwise_conv2d-time-MLP   | cuda_980ti | depthwise_conv2d | time     | MLP         | -0.677676 |  110.823      |  1.64923     |  9.43419    |  0.256052    |   9.47168  |   0.00445908 |    0.00595117 |    260.158    |       1.69531  |        0.0716767 |           3083 |
| cuda_980ti-depthwise_conv2d-power-MLP  | cuda_980ti | depthwise_conv2d | power    | MLP         |  0.728165 |   74.0165     | 13.2142      |  0.0882706  |  9.62604     |   9.29573  |   0.00538427 |   81.001      |    244.784    |     150.137    |      156.638     |           3083 |
| cuda_980ti-depthwise_conv2d-memory-MLP | cuda_980ti | depthwise_conv2d | memory   | MLP         |  0.968198 |    0.600343   |  0.0198049   |  0.103308   |  0.00903908  |   5.16081  |   0.00559127 |    0.0974731  |      3.81964  |       0.187668 |        0.106506  |           3083 |
| cuda_980ti-dilated_conv2d-time-xgb     | cuda_980ti | dilated_conv2d   | time     | xgb         |  0.832078 | 1394.01       |  6.33006     |  6.76809    |  0.95444     |   0.486    |   0.00918293 |    0.00617543 |   4092.52     |      26.034    |        0.81623   |           7857 |
| cuda_980ti-dilated_conv2d-power-xgb    | cuda_980ti | dilated_conv2d   | power    | xgb         |  0.970347 |   54.6792     |  5.51811     |  0.0320421  |  3.44275     |   0.461183 |   0.00494748 |   80.871      |    272.678    |     182.092    |      191.538     |           7857 |
| cuda_980ti-dilated_conv2d-memory-xgb   | cuda_980ti | dilated_conv2d   | memory   | xgb         |  0.996225 |    0.361697   |  0.00539378  |  0.0215081  |  0.00182678  |   0.437965 |   0.011934   |    0.0974731  |      3.82013  |       0.210005 |        0.118103  |           7857 |
| cuda_980ti-dilated_conv2d-time-ert     | cuda_980ti | dilated_conv2d   | time     | ert         |  0.856163 | 1757.68       |  5.86141     |  0.519007   |  0.0290322   |   2.47205  |   0.0752312  |    0.00617543 |   4092.52     |      26.034    |        0.81623   |           7857 |
| cuda_980ti-dilated_conv2d-power-ert    | cuda_980ti | dilated_conv2d   | power    | ert         |  0.96403  |   78.9877     |  4.38169     |  0.0246044  |  1.22097     |   2.19459  |   0.0755116  |   80.871      |    272.678    |     182.092    |      191.538     |           7857 |
| cuda_980ti-dilated_conv2d-memory-ert   | cuda_980ti | dilated_conv2d   | memory   | ert         |  0.998477 |    0.204855   |  0.00208923  |  0.00563093 |  0.000170114 |   2.03808  |   0.0627325  |    0.0974731  |      3.82013  |       0.210005 |        0.118103  |           7857 |
| cuda_980ti-dilated_conv2d-time-dTr     | cuda_980ti | dilated_conv2d   | time     | dTr         |  0.725689 | 1546.37       |  6.17453     |  0.360648   |  0.0019855   |   9.08373  |   0.00185537 |    0.00617543 |   4092.52     |      26.034    |        0.81623   |           7857 |
| cuda_980ti-dilated_conv2d-power-dTr    | cuda_980ti | dilated_conv2d   | power    | dTr         |  0.968969 |   92.0064     |  2.79429     |  0.0163368  |  0.478188    |   7.77692  |   0.00185204 |   80.871      |    272.678    |     182.092    |      191.538     |           7857 |
| cuda_980ti-dilated_conv2d-memory-dTr   | cuda_980ti | dilated_conv2d   | memory   | dTr         |  0.984049 |    0.566223   |  0.00405208  |  0.00985601 | -0           |   6.76226  |   0.00182557 |    0.0974731  |      3.82013  |       0.210005 |        0.118103  |           7857 |
| cuda_980ti-dilated_conv2d-time-MLP     | cuda_980ti | dilated_conv2d   | time     | MLP         |  0.700739 | 1931.09       | 16.1881      | 20.1706     |  2.74645     |  30.2697   |   0.0109278  |    0.00617543 |   4092.52     |      26.034    |        0.81623   |           7857 |
| cuda_980ti-dilated_conv2d-power-MLP    | cuda_980ti | dilated_conv2d   | power    | MLP         |  0.832209 |   77.9676     | 14.7412      |  0.0858208  | 10.8365      |  37.5033   |   0.0103604  |   80.871      |    272.678    |     182.092    |      191.538     |           7857 |
| cuda_980ti-dilated_conv2d-memory-MLP   | cuda_980ti | dilated_conv2d   | memory   | MLP         |  0.977115 |    0.735584   |  0.014364    |  0.0560131  |  0.00548481  |  11.0741   |   0.0100419  |    0.0974731  |      3.82013  |       0.210005 |        0.118103  |           7857 |
| cuda_980ti-max_pool2d-time-xgb         | cuda_980ti | max_pool2d       | time     | xgb         |  0.940609 |    6.93696    |  0.701718    |  0.856741   |  0.340632    |   0.385504 |   0.00485629 |    0.00705457 |     34.0025   |       3.69244  |        1.67619   |           3705 |
| cuda_980ti-max_pool2d-power-xgb        | cuda_980ti | max_pool2d       | power    | xgb         |  0.947458 |   33.9513     |  2.42874     |  0.0156288  |  1.40699     |   3.1581   |   0.0404155  |   88.963      |    204.807    |     158.633    |      151.469     |           3705 |
| cuda_980ti-max_pool2d-memory-xgb       | cuda_980ti | max_pool2d       | memory   | xgb         |  0.966521 |    1.06608    |  0.131664    |  0.175496   |  0.0716757   |   1.75914  |   0.0328918  |    0.0974731  |      5.58112  |       1.09312  |        0.579041  |           3705 |
| cuda_980ti-max_pool2d-time-ert         | cuda_980ti | max_pool2d       | time     | ert         |  0.925723 |    9.09063    |  0.680198    |  0.293297   |  0.271748    |   1.19623  |   0.0377091  |    0.00705457 |     34.0025   |       3.69244  |        1.67619   |           3705 |
| cuda_980ti-max_pool2d-power-ert        | cuda_980ti | max_pool2d       | power    | ert         |  0.914435 |   49.8745     |  2.84416     |  0.0188302  |  1.5569      |   1.01513  |   0.0421599  |   88.963      |    204.807    |     158.633    |      151.469     |           3705 |
| cuda_980ti-max_pool2d-memory-ert       | cuda_980ti | max_pool2d       | memory   | ert         |  0.96049  |    1.4652     |  0.120153    |  0.119905   |  0.0493082   |   1.23069  |   0.0374845  |    0.0974731  |      5.58112  |       1.09312  |        0.579041  |           3705 |
| cuda_980ti-max_pool2d-time-dTr         | cuda_980ti | max_pool2d       | time     | dTr         |  0.755771 |   13.0644     |  1.23072     |  0.441546   |  0.418422    |   1.07832  |   0.00159925 |    0.00705457 |     34.0025   |       3.69244  |        1.67619   |           3705 |
| cuda_980ti-max_pool2d-power-dTr        | cuda_980ti | max_pool2d       | power    | dTr         |  0.844073 |   63.2315     |  3.60388     |  0.0236173  |  1.70087     |   0.9511   |   0.00142097 |   88.963      |    204.807    |     158.633    |      151.469     |           3705 |
| cuda_980ti-max_pool2d-memory-dTr       | cuda_980ti | max_pool2d       | memory   | dTr         |  0.87448  |    2.40167    |  0.237376    |  0.208807   |  0.0852661   |   1.00647  |   0.00141555 |    0.0974731  |      5.58112  |       1.09312  |        0.579041  |           3705 |
| cuda_980ti-max_pool2d-time-MLP         | cuda_980ti | max_pool2d       | time     | MLP         |  0.986747 |    4.99682    |  0.287846    |  0.305572   |  0.125622    |   8.59111  |   0.00633419 |    0.00705457 |     34.0025   |       3.69244  |        1.67619   |           3705 |
| cuda_980ti-max_pool2d-power-MLP        | cuda_980ti | max_pool2d       | power    | MLP         |  0.905163 |   38.2805     |  3.97293     |  0.0257158  |  2.8289      |  22.4557   |   0.00612551 |   88.963      |    204.807    |     158.633    |      151.469     |           3705 |
| cuda_980ti-max_pool2d-memory-MLP       | cuda_980ti | max_pool2d       | memory   | MLP         |  0.98993  |    0.998564   |  0.062853    |  0.0912267  |  0.0346703   |   5.84193  |   0.00944757 |    0.0974731  |      5.58112  |       1.09312  |        0.579041  |           3705 |