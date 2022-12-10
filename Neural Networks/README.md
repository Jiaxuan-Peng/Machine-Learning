#initial weights based on guassian
width = [5, 10, 25, 50, 100]
for i in width:    
    network = initialize_weights1(num_in, i, num_out)
    SGD(network, x,y, 0.02, 0.01, 100, num_out)
    error=0
    for i in range(len(x)):
        row = x[i]
        prediction = predict(network, row)
        error += abs(y[i]-prediction)
    print(error/len(y))
    
#initial weights with 0
width = [5, 10, 25, 50, 100]
for i in width:    
    network = initialize_weights2(num_in, i, num_out)
    SGD(network, x,y, 0.02, 0.01, 100, num_out)
    error=0
    for i in range(len(x)):
        row = x[i]
        prediction = predict(network, row)
        error += abs(y[i]-prediction)
    print(error/len(y))
