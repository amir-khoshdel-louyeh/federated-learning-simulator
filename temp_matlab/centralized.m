function acc = train_centralized(x_train, y_train, x_test, y_test)
% Train centralized model for 3 epochs, batch size 64. Returns accuracy.
    model = make_model();
    model.fit(x_train, y_train, 'epochs', 3, 'batchSize', 64);
    [~, acc] = model.evaluate(x_test, y_test);
end
