from model import make_model


def train_centralized(x_train, y_train, x_test, y_test):
	model = make_model()
	# make_model already returns a compiled model
	model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
	loss, acc = model.evaluate(x_test, y_test, verbose=0)
	return float(acc)

