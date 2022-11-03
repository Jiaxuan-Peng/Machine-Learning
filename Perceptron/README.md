#Standard_Perceptron
w_standard = Standard_Perceptron(x_train, y_train, learn_rate = 0.01, epochs = 10)
error_standard = Prediction_Standard (x_test, y_test, w_standard)

#voted_Perceptron
w_voted, c_voted = voted_Perceptron(x_train, y_train, learn_rate = 0.01, epochs = 10)
error_voted = Prediction_voted(x_test, y_test, w_voted, c_voted)

#Average_Perceptron
w_average = Average_Perceptron(x_train, y_train, learn_rate = 0.01, epochs = 10)
error_average =  Prediction_Average(x_test, y_test, w_average)
