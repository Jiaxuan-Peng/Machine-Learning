For the bank dataset, the related .py files require a long time to obtain the result.

For the credit dataset, unfortunately, there is a bug and I fail to get the result.

#adaboost.py
    DT, alphas = fit(train, 'EP', x_dic, labels, T)##(dataset, gain, x_dic, labels, max_dep, T)
    h_tr = pred(train, DT, alphas)
    h_te = pred(test, DT, alphas)
    
#bagging.py
    trees = fit(train, 'EP', x_dic, labels, 1e+8, T)#(dataset, gain, x_dic, labels, max_dep, T)
    h_tr = pred(train, trees)
    h_te = pred(test, trees)
    
#randomforecast.py
    trees = fit(train, 'EP', x_dic, labels, 1e+8, T, 2)#(dataset, gain, x_dic, labels, max_dep, T, size)
    h_tr = pred(train, trees)
    h_te = pred(test, trees)
