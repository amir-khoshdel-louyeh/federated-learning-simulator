function acc = fednova(x_train, y_train, x_test, y_test, varargin)
% FedNova: Normalized Averaging for Heterogeneous Client Updates.
% Optional name-value:
%  'clients' (3), 'rounds' (3), 'local_epochs' (1), 'batch_size' (64),
%  'full_data_per_client' (false), 'shards' ([]), 'report' (function)

    p = inputParser;
    addParameter(p, 'clients', 3);
    addParameter(p, 'rounds', 3);
    addParameter(p, 'local_epochs', 1);
    addParameter(p, 'batch_size', 64);
    addParameter(p, 'full_data_per_client', false);
    addParameter(p, 'shards', []);
    addParameter(p, 'report', []);
    parse(p, varargin{:});

    clients = p.Results.clients;
    rounds = p.Results.rounds;
    local_epochs = p.Results.local_epochs;
    batch_size = p.Results.batch_size;
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

    for rnd = 1:rounds
        start_time = tic;
        client_deltas_tau = {}; % { {delta}, tau }
        for s = 1:numel(shards)
            shard = shards{s};
            if isempty(shard), continue; end
            local = make_model();
            local.set_weights(global_weights);
            % Count steps
            num_batches = ceil(numel(shard) / batch_size);
            tau_i = num_batches * local_epochs;
            local.fit(x_train(shard,:,:), y_train(shard), 'epochs', local_epochs, 'batchSize', batch_size);
            w_local = local.get_weights();
            delta = cell(size(w_local));
            for j = 1:numel(w_local)
                delta{j} = global_weights{j} - w_local{j};
            end
            client_deltas_tau{end+1} = {delta, tau_i}; %#ok<AGROW>
        end
        if isempty(client_deltas_tau), break; end
        total_tau = 0;
        for k = 1:numel(client_deltas_tau)
            total_tau = total_tau + client_deltas_tau{k}{2};
        end
        weighted_delta = cell(size(global_weights));
        for j = 1:numel(global_weights)
            weighted_delta{j} = zeros(size(global_weights{j}));
        end
        for k = 1:numel(client_deltas_tau)
            delta = client_deltas_tau{k}{1};
            tau_i = client_deltas_tau{k}{2};
            for j = 1:numel(delta)
                weighted_delta{j} = weighted_delta{j} + (tau_i / total_tau) * delta{j};
            end
        end
        for j = 1:numel(global_weights)
            global_weights{j} = global_weights{j} - weighted_delta{j};
        end
        global_model.set_weights(global_weights);

        % Extract deltas only for metrics
        deltas_only = cellfun(@(x) x{1}, client_deltas_tau, 'UniformOutput', false);
        [metrics, prev_acc, prev_global_weights, comm_cost] = compute_round_metrics('FedNova', rnd, start_time, global_model, x_test, y_test, prev_acc, prev_global_weights, global_weights, clients, comm_cost, 'client_deltas', deltas_only);
        if ~isempty(report), report('FedNova', rnd, metrics); end
    end

    global_model.set_weights(global_weights);
    [~, acc] = global_model.evaluate(x_test, y_test);
end

function out = copy_weights(weights)
    out = cell(size(weights));
    for i = 1:numel(weights)
        out{i} = weights{i};
    end
end
