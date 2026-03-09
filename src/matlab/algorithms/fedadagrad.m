function acc = fedadagrad(x_train, y_train, x_test, y_test, varargin)
% FedAdagrad: Server-side Adagrad optimizer.
% Optional name-value:
%  'clients' (3), 'rounds' (3), 'local_epochs' (1), 'batch_size' (64),
%  'server_lr' (0.01), 'tau' (1e-3), 'full_data_per_client' (false), 'shards' ([]), 'report' (function)

    p = inputParser;
    addParameter(p, 'clients', 3);
    addParameter(p, 'rounds', 3);
    addParameter(p, 'local_epochs', 1);
    addParameter(p, 'batch_size', 64);
    addParameter(p, 'server_lr', 0.01);
    addParameter(p, 'tau', 1e-3);
    addParameter(p, 'full_data_per_client', false);
    addParameter(p, 'shards', []);
    addParameter(p, 'report', []);
    parse(p, varargin{:});

    clients = p.Results.clients;
    rounds = p.Results.rounds;
    local_epochs = p.Results.local_epochs;
    batch_size = p.Results.batch_size;
    server_lr = p.Results.server_lr;
    tau = p.Results.tau;
    full_data_per_client = p.Results.full_data_per_client;
    shards = p.Results.shards;
    report = p.Results.report;

    n = size(x_train, 1);
    if isempty(shards)
        shards = make_shards(n, clients, full_data_per_client);
    end

    global_model = make_model();
    global_weights = global_model.get_weights();
    comm_cost = 0;
    prev_acc = [];
    prev_global_weights = copy_weights(global_weights);

    v_t = cell(size(global_weights));
    for j = 1:numel(global_weights)
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
        delta = cell(size(global_weights));
        for j = 1:numel(global_weights)
            delta{j} = global_weights{j} - avg_weights{j};
        end
        new_global_weights = cell(size(global_weights));
        for j = 1:numel(global_weights)
            v_new = v_t{j} + delta{j}.^2;
            new_global_weights{j} = global_weights{j} - server_lr * delta{j} ./ (sqrt(v_new) + tau);
            v_t{j} = v_new;
        end
        global_weights = new_global_weights;
        global_model.set_weights(global_weights);

        [metrics, prev_acc, prev_global_weights, comm_cost] = compute_round_metrics('FedAdagrad', t, start_time, global_model, x_test, y_test, prev_acc, prev_global_weights, global_weights, clients, comm_cost, 'local_weights_list', client_weights, 'reference_weights', avg_weights);
        if ~isempty(report), report('FedAdagrad', t, metrics); end
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

function out = copy_weights(weights)
    out = cell(size(weights));
    for i = 1:numel(weights), out{i} = weights{i}; end
end
