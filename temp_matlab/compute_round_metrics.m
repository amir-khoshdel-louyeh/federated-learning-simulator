function [metrics, prev_acc, prev_global_weights, comm_cost] = compute_round_metrics(algo_name, round_index, start_time, global_model, x_test, y_test, prev_acc, prev_global_weights, global_weights, clients, comm_cost, varargin)
% Compute per-round metrics: accuracy, convergence, communication cost,
% stability/variance, training time, velocity. Mirrors Python metrics.
%
% Optional name-value:
%  'local_weights_list' : cell array of client weight cells
%  'reference_weights'  : cell array weights for stability baseline
%  'client_deltas'      : cell array of client delta cells

    p = inputParser;
    addParameter(p, 'local_weights_list', {});
    addParameter(p, 'reference_weights', {});
    addParameter(p, 'client_deltas', {});
    parse(p, varargin{:});
    local_weights_list = p.Results.local_weights_list;
    reference_weights = p.Results.reference_weights;
    client_deltas = p.Results.client_deltas;

    % Evaluate accuracy
    [~, acc] = global_model.evaluate(x_test, y_test);

    % Convergence
    if isempty(prev_acc)
        convergence = 0.0;
    else
        convergence = acc - prev_acc;
    end

    % Stability / Variance
    if ~isempty(local_weights_list) && ~isempty(reference_weights)
        stability_var = stability_variance_from_weights(local_weights_list, reference_weights);
    elseif ~isempty(client_deltas)
        stability_var = stability_variance_from_deltas(client_deltas);
    else
        stability_var = 0.0;
    end

    % Velocity
    if isempty(prev_global_weights)
        vel = 0.0;
    else
        vel = velocity(prev_global_weights, global_weights);
    end

    % Communication cost (cumulative)
    comm_cost = communication_cost_update(comm_cost, clients, global_weights);

    % Training time
    train_time = toc(start_time);

    metrics = struct();
    metrics.accuracy = acc;
    metrics.convergence = convergence;
    metrics.communication_cost = comm_cost;
    metrics.stability_variance = stability_var;
    metrics.training_time = train_time;
    metrics.velocity = vel;

    prev_acc = acc;
    prev_global_weights = copy_weights(global_weights);
end

function sz = weight_bytes(weights)
    sz = 0;
    for i = 1:numel(weights)
        w = weights{i};
        sz = sz + numel(w) * 8; % assume double precision
    end
end

function comm_cost = communication_cost_update(prev_cost, clients, weights)
    bytes_per_round = clients * 2 * weight_bytes(weights);
    comm_cost = prev_cost + bytes_per_round;
end

function v = stability_variance_from_weights(local_weights_list, reference_weights)
    if isempty(local_weights_list)
        v = 0.0; return;
    end
    norms = zeros(numel(local_weights_list), 1);
    for k = 1:numel(local_weights_list)
        wl = local_weights_list{k};
        ns = 0.0;
        for j = 1:numel(wl)
            diff = wl{j} - reference_weights{j};
            ns = ns + norm(diff(:));
        end
        norms(k) = ns;
    end
    v = var(norms);
end

function v = stability_variance_from_deltas(client_deltas)
    if isempty(client_deltas)
        v = 0.0; return;
    end
    norms = zeros(numel(client_deltas), 1);
    for k = 1:numel(client_deltas)
        delta = client_deltas{k};
        ns = 0.0;
        for j = 1:numel(delta)
            ns = ns + norm(delta{j}(:));
        end
        norms(k) = ns;
    end
    v = var(norms);
end

function vel = velocity(prev_global_weights, global_weights)
    vel = 0.0;
    for j = 1:numel(prev_global_weights)
        vel = vel + norm(global_weights{j}(:) - prev_global_weights{j}(:));
    end
end

function out = copy_weights(weights)
    out = cell(size(weights));
    for i = 1:numel(weights)
        out{i} = weights{i};
    end
end