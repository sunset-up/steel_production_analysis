### MLP 训练日志：

1. activation=relu
   alpha=0.0001
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(100, 50, 25)
   learning_rate=constant
   solver=adam
   MLP:
   Training set:{'rmse': 0.0763, 'mae': 0.0588, 'r2': 0.3295}
   Validation set:{'rmse': 0.074, 'mae': 0.0583, 'r2': 0.3221}
2. activation=relu
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(200, 150, 100, 50, 25)
   learning_rate=constant
   solver=adam
   MLP:
   Training set:{'rmse': 0.0731, 'mae': 0.0557, 'r2': 0.3853}
   Validation set:{'rmse': 0.0711, 'mae': 0.0551, 'r2': 0.3751}
3. activation=relu
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(200, 150, 100, 50, 25)
   learning_rate=adaptive
   solver=adam
   MLP:
   Training set:{'rmse': 0.0803, 'mae': 0.0626, 'r2': 0.2583}
   Validation set:{'rmse': 0.0776, 'mae': 0.062, 'r2': 0.2553}
4. activation=identity
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(200, 150, 100, 50, 25)
   learning_rate=constant
   solver=adam
   MLP:
   Training set:{'rmse': 0.0809, 'mae': 0.0615, 'r2': 0.2465}
   Validation set:{'rmse': 0.0777, 'mae': 0.0605, 'r2': 0.2535}
5. activation=relu
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(200, 150, 100, 50, 25)
   learning_rate=constant
   max_iter=500
   solver=adam
   MLP:
   Training set:{'rmse': 0.0765, 'mae': 0.0577, 'r2': 0.3264}
   Validation set:{'rmse': 0.0744, 'mae': 0.0564, 'r2': 0.3148}
6. activation=relu
   alpha=0.001
   batch_size=auto
   hidden_layer_sizes=(200, 150, 100, 50, 25)
   learning_rate=constant
   max_iter=200
   solver=adam
   MLP:
   Training set:{'rmse': 0.0769, 'mae': 0.0597, 'r2': 0.3193}
   Validation set:{'rmse': 0.0748, 'mae': 0.0593, 'r2': 0.3074}
7. activation=relu
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(300, 250, 200, 150, 100)
   learning_rate=constant
   max_iter=200
   solver=adam
   MLP:
   Training set:{'rmse': 0.0716, 'mae': 0.0539, 'r2': 0.4102}
   Validation set:{'rmse': 0.0705, 'mae': 0.0538, 'r2': 0.3843}
8. activation=relu
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(500, 250, 200, 150, 100)
   learning_rate=constant
   max_iter=200
   solver=adam
   MLP:
   Training set:{'rmse': 0.0713, 'mae': 0.0542, 'r2': 0.4138}
   Validation set:{'rmse': 0.0695, 'mae': 0.0535, 'r2': 0.403}
9. activation=relu
   alpha=0.0001
   batch_size=auto
   hidden_layer_sizes=(500, 400, 300, 150, 100)
   learning_rate=constant
   max_iter=200
   solver=adam
   MLP:
   Training set:{'rmse': 0.0732, 'mae': 0.0555, 'r2': 0.3829}
   Validation set:{'rmse': 0.0711, 'mae': 0.0544, 'r2': 0.3752}
10. activation=relu
    alpha=0.0001
    batch_size=auto
    hidden_layer_sizes=(630, 420, 210, 105, 63)
    learning_rate=constant
    max_iter=200
    solver=adam
    MLP:
    Training set:{'rmse': 0.0715, 'mae': 0.0545, 'r2': 0.411}
    Validation set:{'rmse': 0.0702, 'mae': 0.0543, 'r2': 0.3911}
11. activation=relu
    alpha=0.0001
    batch_size=auto
    hidden_layer_sizes=(210, 168, 126, 84, 42, 21)
    learning_rate=constant
    max_iter=200
    solver=adam
    MLP:
    Training set:{'rmse': 0.079, 'mae': 0.0624, 'r2': 0.2821}
    Validation set:{'rmse': 0.0772, 'mae': 0.0624, 'r2': 0.2619}
12. activation=relu
    alpha=0.0001
    batch_size=auto
    hidden_layer_sizes=(252, 168, 126, 84, 42)
    learning_rate=constant
    max_iter=300
    solver=adam
    MLP:
    Training set:{'rmse': 0.0749, 'mae': 0.0566, 'r2': 0.3547}
    Validation set:{'rmse': 0.0736, 'mae': 0.0561, 'r2': 0.3305}
13. activation=relu
    alpha=0.0001
    batch_size=auto
    hidden_layer_sizes=(210, 168, 126, 84)
    learning_rate=constant
    max_iter=200
    solver=adam
    MLP:
    Training set:{'rmse': 0.0733, 'mae': 0.0553, 'r2': 0.3806}
    Validation set:{'rmse': 0.0713, 'mae': 0.0543, 'r2': 0.3708}
14. activation=relu
    alpha=0.0001
    batch_size=auto
    hidden_layer_sizes=(252, 168, 126, 84)
    learning_rate=constant
    max_iter=200
    solver=adam
    MLP:
    Training set:{'rmse': 0.0733, 'mae': 0.0553, 'r2': 0.3813}
    Validation set:{'rmse': 0.0714, 'mae': 0.0548, 'r2': 0.3694}
15. activation=relu
    alpha=0.0001
    batch_size=auto
    hidden_layer_sizes=(420, 168, 126, 84, 42)
    learning_rate=constant
    max_iter=400
    solver=adam
    MLP:
    Training set:{'rmse': 0.0756, 'mae': 0.0577, 'r2': 0.3412}
    Validation set:{'rmse': 0.0739, 'mae': 0.0575, 'r2': 0.3235}
16. activation=relu
    alpha=0.001
    batch_size=auto
    hidden_layer_sizes=(210, 168, 126, 84)
    learning_rate=constant
    max_iter=400
    solver=adam
    MLP:
    Training set:{'rmse': 0.0746, 'mae': 0.0577, 'r2': 0.3597}
    Validation set:{'rmse': 0.0729, 'mae': 0.0577, 'r2': 0.3426}
17. activation=relu
    alpha=0.01
    batch_size=auto
    hidden_layer_sizes=(210, 168, 126, 84)
    learning_rate=constant
    max_iter=400
    solver=adam
    MLP:
    Training set:{'rmse': 0.0724, 'mae': 0.0552, 'r2': 0.3968}
    Validation set:{'rmse': 0.0702, 'mae': 0.0548, 'r2': 0.3911}
18. activation=relu
    alpha=0.01
    batch_size=100
    hidden_layer_sizes=(210, 168, 126, 84)
    learning_rate=constant
    max_iter=400
    solver=adam
    MLP:
    Training set:{'rmse': 0.0734, 'mae': 0.0559, 'r2': 0.3794}
    Validation set:{'rmse': 0.0703, 'mae': 0.0545, 'r2': 0.3888}
19. activation=relu
    alpha=0.01
    batch_size=auto
    hidden_layer_sizes=(210, 168, 126, 84)
    learning_rate=constant
    max_iter=200
    solver=adam
    MLP:
    Training set:{'rmse': 0.0724, 'mae': 0.0552, 'r2': 0.3968}
    Validation set:{'rmse': 0.0702, 'mae': 0.0548, 'r2': 0.3911}
20. activation=relu
    alpha=0.01
    batch_size=auto
    hidden_layer_sizes=(420, 336, 252, 168)
    learning_rate=constant
    max_iter=200
    solver=adam
    MLP:
    Training set:{'rmse': 0.0716, 'mae': 0.0541, 'r2': 0.41}
    Validation set:{'rmse': 0.0692, 'mae': 0.053, 'r2': 0.4072}
21. activation=relu
    alpha=0.01
    batch_size=auto
    hidden_layer_sizes=(420, 336, 252, 168, 84)
    learning_rate=constant
    max_iter=200
    solver=adam
    MLP:
    Training set:{'rmse': 0.0718, 'mae': 0.0545, 'r2': 0.4059}
    Validation set:{'rmse': 0.0696, 'mae': 0.0537, 'r2': 0.4008}