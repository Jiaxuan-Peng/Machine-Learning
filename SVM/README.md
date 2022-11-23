#SVM_primal_gamma1
w1 = SVM_primal_gamma1(x_train, y_train, C, gamma=0.01,T=100)
error_train = Prediction_SVM_primal (x_train, y_train, w2)

#SVM_primal_gamma2 
w2 = SVM_primal_gamma2(x_train, y_train, C, gamma=0.01,T=100)
error_train = Prediction_SVM_primal (x_train, y_train, w2)

#SVM_dual
w,b = SVM_dual(x_train, y_train, C)
error_train = Prediction_SVM_dual (x_train, y_train, w,b)

#SVM_dual_kernel
alpha,b = SVM_dual_kernel(x_train, y_train, C,gamma)
error_train = Prediction_SVM_dual_kernel (x_train, y_train, alpha,b)

#kernel_Perceptron
w_kernel, c_kernel = kernel_Perceptron(x_train, y_train, 0.01, 10,gamma)
error_kernel = Prediction_kernel(x_test, y_test, w_kernel, c_kernel)
