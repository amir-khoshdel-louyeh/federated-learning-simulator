function acc = fedadam(x_train, y_train, x_test, y_test, varargin)
% FedAdam: Server-side Adam optimizer for global aggregation.
% Optional name-value:
%  'clients' (3), 'rounds' (3), 'local_epochs' (1), 'batch_size' (64),
%  'server_lr' (0.01), 'beta1' (0.9), 'beta2' (0.999), 'tau' (1e-3),
%  'full_data_per_client' (false), 'report' (function)

    p = inputParser;
    addParameter(p, 'clients', 3);
    addParameter(p, 'rounds', 3);
    addParameter(p, 'local_epochs', 1);
    addParameter(p, 'batch_size', 64);
    addParameter(p, 'server_lr', 0.01);
    addParameter(p, 'beta1', 0.9);
    addParameter(p, 'beta2', 0.999);
    addParameter(p, 'tau', 1e-3);
    addParameter(p, 'full_data_per_client', false);
    addParameter(p, 'report', []);
    parse(p, varargin{:});

    clients = p.Results.clients;
    rounds = p.Results.rounds;
    local_epochs = p.Results.local_epochs;
    batch_size = p.Results.batch_size;
    server_lr = p.Results.server_lr;
    beta1 = p.Results.beta1;
    beta2 = p.Results.beta2;
    tau = p.Results.tau;
    full_data_per_client = p.Results.full_data_per_client;
    report = p.Results.report;

    n = size(x_train, 1);
    shards = make_shards(n, clients, full_data_per_client);

    global_model = make_model();
    global_weights = global_model.get_weights();
    comm_cost = 0;
    prev_acc = [];
    prev_global_weights = copy_weights(global_weights);

    % Server moments
    m_t = cell(size(global_weights));
    v_t = cell(size(global_weights));
    for j = 1:numel(global_weights)
        m_t{j} = zeros(size(global_weights{j}));
        v_t{j} = zeros(size(global_weights{j}));
    end

    for t = 1:rounds
        start_time = tic;
        client_weights = {};
        for s = 1:numel(shards)
            shard = shards{s}; if isempty(shard), continue; end
            local = make_model();
            local.set_weights(global_weights);
            local.fit(x_train(shard,:,:), y_train(shard), 'epochs', local_epochs, 'batchSize', batch_size);
            client_weights{end+1} = local.get_weights(); %#ok<AGROW>
        end
        if isempty(client_weights), break; end
        avg_weights = average_weights(client_weights);
        % Pseudo-gradient
        delta = cell(size(global_weights));
        for j = 1:numel(global_weights)
            delta{j} = global_weights{j} - avg_weights{j};
        end
        % Adam update on server
        new_global_weights = cell(size(global_weights));
        for j = 1:numel(global_weights)
            m_new = beta1 * m_t{j} + (1 - beta1) * delta{j};
            v_new = beta2 * v_t{j} + (1 - beta2) * (delta{j}.^2);
            m_hat = m_new ./ (1 - beta1^t);
            v_hat = v_new ./ (1 - beta2^t);
            new_global_weights{j} = global_weights{j} - server_lr * m_hat ./ (sqrt(v_hat) + tau);
            m_t{j} = m_new; v_t{j} = v_new;
        end
        global_weights = new_global_weights;
        global_model.set_weights(global_weights);

        [metrics, prev_acc, prev_global_weights, comm_cost] = compute_round_metrics('FedAdam', t, start_time, global_model, x_test, y_test, prev_acc, prev_global_weights, global_weights, clients, comm_cost, 'local_weights_list', client_weights, 'reference_weights', avg_weights);
        if ~isempty(report), report('FedAdam', t, metrics); end
    end

    global_model.set_weights(global_weights);
    [~, acc] = global_model.evaluate(x_test, y_test);
end

function new_weights = average_weights(client_weights)
    K = numel(client_weights);
    L = numel(client_weights{1});
    new_weights = cell(1, L);
    for j = 1:L
        sumW = 0; for k = 1:K, sumW = sumW + client_weights{k}{j}; end
        new_weights{j} = sumW / K;
    end
end

function out = copy_weights(weights)
    out = cell(size(weights));
    for i = 1:numel(weights), out{i} = weights{i}; end
end
