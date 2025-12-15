function model = make_model()
% make_model: Returns a simple MLP model for MNIST with manual training using Adam.
% Architecture: Flatten(28x28) -> Dense(64, ReLU) -> Dense(10, Softmax)
% Provides methods: fit, evaluate, get_weights, set_weights.
%
% No toolboxes required; backprop is implemented manually.

    % Hyperparameters
    model.lr = 0.001;           % local optimizer learning rate (Adam)
    model.beta1 = 0.9;          % Adam beta1
    model.beta2 = 0.999;        % Adam beta2
    model.eps = 1e-8;           % Adam epsilon
    model.batchSizeDefault = 64;

    % Initialize weights
    in_dim = 28 * 28;
    hid_dim = 64;
    out_dim = 10;
    rng('default');
    scale1 = sqrt(2 / in_dim);
    scale2 = sqrt(2 / hid_dim);
    model.W1 = scale1 * randn(in_dim, hid_dim);
    model.b1 = zeros(1, hid_dim);
    model.W2 = scale2 * randn(hid_dim, out_dim);
    model.b2 = zeros(1, out_dim);

    % Adam optimizer state
    model.t = 0;
    model.mW1 = zeros(size(model.W1));
    model.mb1 = zeros(size(model.b1));
    model.mW2 = zeros(size(model.W2));
    model.mb2 = zeros(size(model.b2));

    model.vW1 = zeros(size(model.W1));
    model.vb1 = zeros(size(model.b1));
    model.vW2 = zeros(size(model.W2));
    model.vb2 = zeros(size(model.b2));

    % Methods
    model.fit = @fit;
    model.evaluate = @evaluate;
    model.predict = @predict;
    model.get_weights = @get_weights;
    model.set_weights = @set_weights;

    function yprob = predict(X)
        % X: N x 28 x 28 or N x 784
        if ndims(X) == 3
            X = reshape(X, size(X,1), []);
        end
        Z1 = X * model.W1 + repmat(model.b1, size(X,1), 1);
        A1 = max(0, Z1); % ReLU
        Z2 = A1 * model.W2 + repmat(model.b2, size(A1,1), 1);
        yprob = softmax_rows(Z2);
    end

    function [loss, acc] = evaluate(X, y)
        % y: N x 1 integer labels in [0..9]
        P = predict(X);
        N = size(P,1);
        yIdx = y(:) + 1; % MATLAB 1-based
        yIdx(yIdx < 1) = 1; yIdx(yIdx > 10) = 10;
        p_true = P(sub2ind(size(P), (1:N)', yIdx));
        loss = -mean(log(max(p_true, 1e-12)));
        [~, pred] = max(P, [], 2);
        acc = mean((pred-1) == y(:));
    end

    function fit(X, y, varargin)
        % fit(X, y, 'epochs', E, 'batchSize', B, 'muProx', 0, 'global_weights', [])
        % Optional proximal term: mu/2 * ||w - w_global||^2
        p = inputParser;
        addParameter(p, 'epochs', 1);
        addParameter(p, 'batchSize', model.batchSizeDefault);
        addParameter(p, 'muProx', 0.0);
        addParameter(p, 'global_weights', []);
        parse(p, varargin{:});
        epochs = p.Results.epochs;
        batchSize = p.Results.batchSize;
        muProx = p.Results.muProx;
        gw = p.Results.global_weights; % cell {W1,b1,W2,b2}
        useProx = (~isempty(gw)) && (muProx > 0);

        N = size(X,1);
        if ndims(X) == 3
            X = reshape(X, N, []);
        end
        Y = y(:);
        for ep = 1:epochs
            idx = randperm(N);
            for s = 1:batchSize:N
                batch_idx = idx(s:min(s+batchSize-1, N));
                Xb = X(batch_idx, :);
                yb = Y(batch_idx);
                % Forward
                Z1 = Xb * model.W1 + repmat(model.b1, size(Xb,1), 1);
                A1 = max(0, Z1);
                Z2 = A1 * model.W2 + repmat(model.b2, size(A1,1), 1);
                P = softmax_rows(Z2);
                % Loss grad w.r.t logits
                Yoh = onehot(yb, out_dim);
                dZ2 = (P - Yoh) / size(Xb,1);
                % Gradients for second layer
                dW2 = A1' * dZ2;
                db2 = sum(dZ2, 1);
                % Backprop to first layer
                dA1 = dZ2 * model.W2';
                dZ1 = dA1;
                dZ1(Z1 <= 0) = 0; % ReLU derivative
                dW1 = Xb' * dZ1;
                db1 = sum(dZ1, 1);
                % Proximal term grads if applicable
                if useProx
                    dW2 = dW2 + muProx * (model.W2 - gw{3});
                    db2 = db2 + muProx * (model.b2 - gw{4});
                    dW1 = dW1 + muProx * (model.W1 - gw{1});
                    db1 = db1 + muProx * (model.b1 - gw{2});
                end
                % Adam update
                model.t = model.t + 1;
                [model.W2, model.mW2, model.vW2] = adam_step(model.W2, dW2, model.mW2, model.vW2, model.t, model.lr, model.beta1, model.beta2, model.eps);
                [model.b2, model.mb2, model.vb2] = adam_step(model.b2, db2, model.mb2, model.vb2, model.t, model.lr, model.beta1, model.beta2, model.eps);
                [model.W1, model.mW1, model.vW1] = adam_step(model.W1, dW1, model.mW1, model.vW1, model.t, model.lr, model.beta1, model.beta2, model.eps);
                [model.b1, model.mb1, model.vb1] = adam_step(model.b1, db1, model.mb1, model.vb1, model.t, model.lr, model.beta1, model.beta2, model.eps);
            end
        end
    end

    function weights = get_weights()
        weights = {model.W1, model.b1, model.W2, model.b2};
    end

    function set_weights(weights)
        model.W1 = weights{1};
        model.b1 = weights{2};
        model.W2 = weights{3};
        model.b2 = weights{4};
    end

end

function Y = onehot(y, K)
    y = y(:);
    N = numel(y);
    Y = zeros(N, K);
    for i = 1:N
        k = y(i) + 1; % 1-based
        if k < 1, k = 1; end
        if k > K, k = K; end
        Y(i, k) = 1;
    end
end

function P = softmax_rows(Z)
    Z = Z - max(Z, [], 2); % for numerical stability
    E = exp(Z);
    S = sum(E, 2);
    P = E ./ S;
end

function [w_new, m_new, v_new] = adam_step(w, g, m, v, t, lr, beta1, beta2, eps)
    m_new = beta1 * m + (1 - beta1) * g;
    v_new = beta2 * v + (1 - beta2) * (g.^2);
    m_hat = m_new ./ (1 - beta1^t);
    v_hat = v_new ./ (1 - beta2^t);
    w_new = w - lr .* m_hat ./ (sqrt(v_hat) + eps);
end