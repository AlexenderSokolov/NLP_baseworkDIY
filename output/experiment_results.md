| feature     | n_list   | loss             |   learning_rate |   train_accuracy |   validate_accuracy |   test_accuracy |
|:------------|:---------|:-----------------|----------------:|-----------------:|--------------------:|----------------:|
| BoW         | -        | cross_entropy_ls |          0.001  |         0.890355 |            0.485932 |        0.488667 |
| NgramFusion | 1,2,3    | cross_entropy_ls |          0.001  |         0.941953 |            0.485932 |        0.481717 |
| BoW         | -        | cross_entropy    |          0.001  |         0.885078 |            0.48476  |        0.488969 |
| NgramFusion | 1,2,3    | cross_entropy    |          0.001  |         0.972442 |            0.48476  |        0.480206 |
| NgramFusion | 1,2,3    | cross_entropy    |          0.0005 |         0.885078 |            0.483001 |        0.484134 |
| NgramFusion | 1,2,3    | cross_entropy_ls |          0.0005 |         0.947962 |            0.483001 |        0.47809  |
| BoW         | -        | cross_entropy_ls |          0.0005 |         0.863383 |            0.48007  |        0.492596 |
| BoW         | -        | cross_entropy    |          0.0005 |         0.848285 |            0.48007  |        0.490481 |