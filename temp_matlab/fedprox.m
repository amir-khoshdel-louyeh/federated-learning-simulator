function acc = fedprox(x_train, y_train, x_test, y_test, varargin)
% FedProx implementation with proximal term.
% Optional name-value:
%  'clients' (3), 'rounds' (3), 'mu' (0.01), 'local_epochs' (1), 'batch_size' (64),
%  'full_data_per_client' (false), 'report' (function)

    p = inputParser;
    addParameter(p, 'clients', 3);
    addParameter(p, 'rounds', 3);
    addParameter(p, 'mu', 0.01);
    addParameter(p, 'local_epochs', 1);
    addParameter(p, 'batch_size', 64);
    addParameter(p, 'full_data_per_client', false);
    addParameter(p, 'report', []);
    parse(p, varargin{:});

    clients = p.Results.clients;
    rounds = p.Results.rounds;
    mu = p.Results.mu;
    local_epochs = p.Results.local_epochs;
    batch_size = p.Results.batch_size;
    full_data_per_client = p.Results.full_data_per_client;
    report = p.Results.report;

    n = size(x_train, 1);
    shards = make_shards(n, clients, full_data_per_client);

    global_model = make_model();
    global_weights = global_model.get_weights();
    comm_cost = 0;
    prev_acc = [];
    prev_global_weights = copy_weights(global_weights);

    for r = 1:rounds
        start_time = tic;
        client_weights = {};
        for s = 1:numel(shards)
            shard = shards{s};
            if isempty(shard), continue; end
            local = make_model();
            local.set_weights(global_weights);
            local.fit(x_train(shard,:,:), y_train(shard), 'epochs', local_epochs, 'batchSize', batch_size, 'muProx', mu, 'global_weights', global_weights);
            client_weights{end+1} = local.get_weights(); %#ok<AGROW>
        end
        if isempty(client_weights), break; end
        new_weights = average_weights(client_weights);
        global_weights = new_weights;
        global_model.set_weights(global_weights);

        [metrics, prev_acc, prev_global_weights, comm_cost] = compute_round_metrics('FedProx', r, start_time, global_model, x_test, y_test, prev_acc, prev_global_weights, global_weights, clients, comm_cost, 'local_weights_list', client_weights, 'reference_weights', global_weights);
        if ~isempty(report), report('FedProx', r, metrics); end
    end

    global_model.set_weights(global_weights);
    [~, acc] = global_model.evaluate(x_test, y_test);
end

function new_weights = average_weights(client_weights)
    K = numel(client_weights);
    L = numel(client_weights{1});
    new_weights = cell(1, L);
    for j = 1:L
        sumW = 0;
        for k = 1:K
            sumW = sumW + client_weights{k}{j};
        end
        new_weights{j} = sumW / K;
    end
end

function out = copy_weights(weights)
    out = cell(size(weights));
    for i = 1:numel(weights)
        out{i} = weights{i};
    end
end
