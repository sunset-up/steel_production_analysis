### GPR训练日志：

1. alpha=0.01
   kernel=Matern(length_scale=10, nu=1)
   GPR:
   Training set:{'rmse': 0.0693, 'mae': 0.0526, 'r2': 0.4465}
   Validation set:{'rmse': 0.0704, 'mae': 0.0543, 'r2': 0.3874}
2. training time: 183.10s
   alpha=0.01
   kernel=RBF(length_scale=1)
   GPR:
   Training set:{'rmse': 0.0714, 'mae': 0.0542, 'r2': 0.4131}
   Validation set:{'rmse': 0.0705, 'mae': 0.0546, 'r2': 0.3852}
3. training time: 1025.55s
   alpha=0.01
   kernel=1**2 * RBF(length_scale=1)
   GPR:
   Training set:{'rmse': 0.0712, 'mae': 0.054, 'r2': 0.4165}
   Validation set:{'rmse': 0.0705, 'mae': 0.0546, 'r2': 0.3852}
4. training time: 350.44s
   alpha=0.01
   kernel=RBF(length_scale=10)
   GPR:
   Training set:{'rmse': 0.0714, 'mae': 0.0542, 'r2': 0.4131}
   Validation set:{'rmse': 0.0705, 'mae': 0.0546, 'r2': 0.3852}
5. training time: 130.01s
   alpha=0.01
   kernel=RBF(length_scale=100)
   GPR:
   Training set:{'rmse': 0.0714, 'mae': 0.0542, 'r2': 0.4131}
   Validation set:{'rmse': 0.0705, 'mae': 0.0546, 'r2': 0.3852}
6. training time: 35.36s
   alpha=0.01
   kernel=RBF(length_scale=1)
   normalize_y=True

   GPR:
   Training set:{'rmse': 0.0021, 'mae': 0.0009, 'r2': 0.9995}
   Validation set:{'rmse': 0.0848, 'mae': 0.0655, 'r2': 0.1103}

   training time: 35.20s
   alpha=0.01
   kernel=RBF(length_scale=1, length_scale_bound=(1, 1000))
   normalize_y=True
   GPR:
   Training set:{'rmse': 0.0021, 'mae': 0.0009, 'r2': 0.9995}
   Validation set:{'rmse': 0.0848, 'mae': 0.0655, 'r2': 0.1103}
7. training time: 35.21s
   alpha=0.05
   kernel=RBF(length_scale=1)
   normalize_y=True
   GPR:
   Training set:{'rmse': 0.0059, 'mae': 0.0041, 'r2': 0.9959}
   Validation set:{'rmse': 0.0846, 'mae': 0.0654, 'r2': 0.1144}
8. training time: 11.56s
   PCA(n_components=8)

   alpha=0.08
   kernel=RBF(length_scale=1)
   normalize_y=True
   GPR:
   Training set:{'rmse': 0.0069, 'mae': 0.0053, 'r2': 0.9945}
   Validation set:{'rmse': 0.09, 'mae': 0.0704, 'r2': -0.0014}
9. training time: 647.64s
   alpha=0.05
   kernel=RBF(length_scale=1) + WhiteKernel(noise_level=1)
   normalize_y=True
   GPR:
   Training set:{'rmse': 0.0158, 'mae': 0.0117, 'r2': 0.9711}
   Validation set:{'rmse': 0.0842, 'mae': 0.0651, 'r2': 0.1223