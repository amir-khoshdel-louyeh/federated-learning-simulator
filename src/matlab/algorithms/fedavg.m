function acc = fedavg(x_train, y_train, x_test, y_test, varargin)
% Federated Averaging (FedAvg) in MATLAB mirroring Python implementation.
% Optional name-value:
%  'clients' (default 3)
%  'rounds' (default 3)
%  'full_data_per_client' (default false)
%  'shards' (default [])
%  'report' (function handle)

    p = inputParser;
    addParameter(p, 'clients', 3);
    addParameter(p, 'rounds', 3);
    addParameter(p, 'full_data_per_client', false);
    addParameter(p, 'shards', []);
    addParameter(p, 'report', []);
    parse(p, varargin{:});

    clients = p.Results.clients;
    rounds = p.Results.rounds;
    full_data_per_client = p.Results.full_data_per_client;
    shards = p.Results.shards;
    report = p.Results.report;

    len_xtrain = size(x_train, 1);
    if isempty(shards)
        shards = make_shards(len_xtrain, clients, full_data_per_client);
    end

    global_model = make_model();
    global_weights = global_model.get_weights();
    comm_cost = 0;
    prev_acc = [];
    prev_global_weights = copy_weights(global_weights);

    for i = 1:rounds
        start_time = tic;
        client_weights = {};
        for s = 1:numel(shards)
            shard = shards{s};
            if isempty(shard), continue; end
            local = make_model();
            local.set_weights(global_weights);
            local.fit(x_train(shard,:,:), y_train(shard), 'epochs', 1, 'batchSize', 64);
            client_weights{end+1} = local.get_weights(); %#ok<AGROW>
        end
        % Average weights
        new_weights = average_weights(client_weights);
        global_weights = new_weights;
        global_model.set_weights(global_weights);

        [metrics, prev_acc, prev_global_weights, comm_cost] = compute_round_metrics('FedAvg', i, start_time, global_model, x_test, y_test, prev_acc, prev_global_weights, global_weights, clients, comm_cost, 'local_weights_list', client_weights, 'reference_weights', global_weights);
        if ~isempty(report)
            report('FedAvg', i, metrics);
        end
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
