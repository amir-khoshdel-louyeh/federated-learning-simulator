function acc = centralized(x_train, y_train, x_test, y_test)
% Centralized training (wrapper), delegates to train_centralized helper.
    acc = train_centralized(x_train, y_train, x_test, y_test);
end
