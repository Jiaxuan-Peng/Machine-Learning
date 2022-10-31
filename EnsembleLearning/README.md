For the bank dataset, the related .py files require a long time to obtain the result.

For the credit dataset, unfortunately, there is a bug and I fail to get the result.

e1, e2 = adaboost(train, test, 'EP', x_dic, labels, 25)

#bagging
    trees = fit(train, 'EP', x_dic, labels, 1e+8, T)
    h_tr = pred(train, trees)
    h_te = pred(test, trees)
