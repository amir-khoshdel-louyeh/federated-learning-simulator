function acc = scaffold(x_train, y_train, x_test, y_test, varargin)
% SCAFFOLD implementation with control variates.
% Optional name-value:
%  'clients' (3), 'rounds' (3), 'local_epochs' (1), 'batch_size' (64), 'lr' (0.001),
%  'full_data_per_client' (false), 'report' (function)

    p = inputParser;
    addParameter(p, 'clients', 3);
    addParameter(p, 'rounds', 3);
    addParameter(p, 'local_epochs', 1);
    addParameter(p, 'batch_size', 64);
    addParameter(p, 'lr', 0.001);
    addParameter(p, 'full_data_per_client', false);
    addParameter(p, 'report', []);
    parse(p, varargin{:});

    clients = p.Results.clients;
    rounds = p.Results.rounds;
    local_epochs = p.Results.local_epochs;
    batch_size = p.Results.batch_size;
    lr = p.Results.lr;
    full_data_per_client = p.Results.full_data_per_client;
    report = p.Results.report;

    n = size(x_train, 1);
    shards = make_shards(n, clients, full_data_per_client);

    global_model = make_model();
    global_model.lr = lr; % set LR
    global_weights = global_model.get_weights();
    comm_cost = 0;
    prev_acc = [];
    prev_global_weights = copy_weights(global_weights);

    % Server control variate c and per-client c_i
    c = cellfun(@(w) zeros(size(w)), global_weights, 'UniformOutput', false);
    c_i_list = repmat({c}, 1, clients);

    out_dim = 10; % fixed for MNIST

    for rnd = 1:rounds
        start_time = tic;
        client_weights = {};
        new_c_i_list = cell(1, clients);

        for client_idx = 1:clients
            shard = shards{client_idx};
            if isempty(shard)
                new_c_i_list{client_idx} = c_i_list{client_idx};
                continue;
            end
            local = make_model();
            local.lr = lr;
            local.set_weights(global_weights);

            % Manual local training with control variate correction
            steps = 0;
            N = numel(shard);
            Xall = x_train(shard,:,:);
            Yall = y_train(shard);
            if ndims(Xall) == 3
                Xall = reshape(Xall, N, []);
            end
            for ep = 1:local_epochs
                perm = randperm(N);
                for s = 1:batch_size:N
                    batch_idx = perm(s:min(s+batch_size-1, N));
                    Xb = Xall(batch_idx, :);
                    yb = Yall(batch_idx);
                    % Forward
                    Z1 = Xb * local.W1 + repmat(local.b1, size(Xb,1), 1);
                    A1 = max(0, Z1);
                    Z2 = A1 * local.W2 + repmat(local.b2, size(A1,1), 1);
                    P = softmax_rows(Z2);
                    Yoh = onehot(yb, out_dim);
                    dZ2 = (P - Yoh) / size(Xb,1);
                    dW2 = A1' * dZ2;
                    db2 = sum(dZ2, 1);
                    dA1 = dZ2 * local.W2';
                    dZ1 = dA1; dZ1(Z1 <= 0) = 0;
                    dW1 = Xb' * dZ1;
                    db1 = sum(dZ1, 1);

                    % Correction (server_c - client_c)
                    corrW1 = c{1} - c_i_list{client_idx}{1};
                    corrb1 = c{2} - c_i_list{client_idx}{2};
                    corrW2 = c{3} - c_i_list{client_idx}{3};
                    corrb2 = c{4} - c_i_list{client_idx}{4};

                    dW1 = dW1 + corrW1;
                    db1 = db1 + corrb1;
                    dW2 = dW2 + corrW2;
                    db2 = db2 + corrb2;

                    % Adam step
                    local.t = local.t + 1;
                    [local.W2, local.mW2, local.vW2] = adam_step(local.W2, dW2, local.mW2, local.vW2, local.t, lr, local.beta1, local.beta2, local.eps);
                    [local.b2, local.mb2, local.vb2] = adam_step(local.b2, db2, local.mb2, local.vb2, local.t, lr, local.beta1, local.beta2, local.eps);
                    [local.W1, local.mW1, local.vW1] = adam_step(local.W1, dW1, local.mW1, local.vW1, local.t, lr, local.beta1, local.beta2, local.eps);
                    [local.b1, local.mb1, local.vb1] = adam_step(local.b1, db1, local.mb1, local.vb1, local.t, lr, local.beta1, local.beta2, local.eps);

                    steps = steps + 1;
                end
            end

            w_local = local.get_weights();
            client_weights{end+1} = w_local; %#ok<AGROW>

            % Update client control variate: c_i' = c_i - c + (1/(steps*lr)) * (w_global - w_local)
            old_ci = c_i_list{client_idx};
            scale = 1.0 / (max(1, steps) * lr);
            new_ci = cell(1, numel(global_weights));
            for j = 1:numel(global_weights)
                new_ci{j} = old_ci{j} - c{j} + scale * (global_weights{j} - w_local{j});
            end
            new_c_i_list{client_idx} = new_ci;
        end

        % Average client weights to form new global weights
        if isempty(client_weights), break; end
        new_weights = average_weights(client_weights);

        % Update server control variate: c = c + (1/K) * sum_i (c_i' - c_i)
        sum_delta = cellfun(@(w) zeros(size(w)), global_weights, 'UniformOutput', false);
        for i = 1:clients
            old_ci = c_i_list{i}; new_ci = new_c_i_list{i};
            for j = 1:numel(global_weights)
                sum_delta{j} = sum_delta{j} + (new_ci{j} - old_ci{j});
            end
        end
        K = clients;
        for j = 1:numel(global_weights)
            c{j} = c{j} + (sum_delta{j} / K);
        end

        global_weights = new_weights;
        c_i_list = new_c_i_list;
        global_model.set_weights(global_weights);

        [metrics, prev_acc, prev_global_weights, comm_cost] = compute_round_metrics('SCAFFOLD', rnd, start_time, global_model, x_test, y_test, prev_acc, prev_global_weights, global_weights, clients, comm_cost, 'local_weights_list', client_weights, 'reference_weights', global_weights);
        if ~isempty(report), report('SCAFFOLD', rnd, metrics); end
    end

    global_model.set_weights(global_weights);
    [~, acc] = global_model.evaluate(x_test, y_test);
end

function new_weights = average_weights(client_weights)
    K = numel(client_weights);
    L = numel(client_weights{1});
    new_weights = cell(1, L);
    for j = 1:L
        sW = 0; for k = 1:K, sW = sW + client_weights{k}{j}; end
        new_weights{j} = sW / K;
    end
end

function Y = onehot(y, K)
    y = y(:); N = numel(y); Y = zeros(N, K);
    for i = 1:N
        k = y(i) + 1; k = max(1, min(K, k));
        Y(i, k) = 1;
    end
end

function P = softmax_rows(Z)
    Z = Z - max(Z, [], 2);
    E = exp(Z); S = sum(E, 2);
    P = E ./ S;
end

function [w_new, m_new, v_new] = adam_step(w, g, m, v, t, lr, beta1, beta2, eps)
    m_new = beta1 * m + (1 - beta1) * g;
    v_new = beta2 * v + (1 - beta2) * (g.^2);
    m_hat = m_new ./ (1 - beta1^t);
    v_hat = v_new ./ (1 - beta2^t);
    w_new = w - lr .* m_hat ./ (sqrt(v_hat) + eps);
end

function out = copy_weights(weights)
    out = cell(size(weights));
    for i = 1:numel(weights), out{i} = weights{i}; end
end
