function main
% MATLAB port of federated-learning-simulator to compare results with Python
% Mirrors Python runtime options except UI format (CLI prompts here).
    % Ensure subfolders (e.g., algorithms) are on the MATLAB path
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    addpath(fullfile(here, 'algorithms'));

    % Parameters (prompt user for inputs to match Python GUI behavior)
    try
        total_dataset = input('Enter total dataset size (default 10000): ');
        if isempty(total_dataset) || ~isnumeric(total_dataset) || total_dataset <= 0
            total_dataset = 10000;
        end
    catch
        total_dataset = 10000;
    end
    try
        clients = input('Enter number of clients (default 3): ');
        if isempty(clients) || ~isnumeric(clients) || clients <= 0
            clients = 3;
        end
    catch
        clients = 3;
    end
    try
        rounds = input('Enter number of rounds (default 3): ');
        if isempty(rounds) || ~isnumeric(rounds) || rounds <= 0
            rounds = 3;
        end
    catch
        rounds = 3;
    end
    % Dataset mode selection (matches Python GUI semantics)
    try
        dataset_mode = strtrim(lower(input('Data distribution [i/n or iid/non_iid] (default iid): ', 's')));
        if isempty(dataset_mode)
            dataset_mode = 'iid';
        end
        % Accept shorthand: 'i' for iid, 'n' for non_iid
        if strcmp(dataset_mode, 'i')
            dataset_mode = 'iid';
        elseif strcmp(dataset_mode, 'n')
            dataset_mode = 'non_iid';
        end
        if ~ismember(dataset_mode, {'iid', 'non_iid'})
            dataset_mode = 'iid';
        end
    catch
        dataset_mode = 'iid';
    end

    common_label = 7;
    common_fraction = 0.1;
    if strcmp(dataset_mode, 'non_iid')
        try
            cl = input('Common label [0..9] (default 7): ');
            if ~isempty(cl) && isnumeric(cl) && cl >= 0 && cl <= 9
                common_label = floor(cl);
            end
        catch
        end
        try
            cf = input('Common fraction [0..1] (default 0.1): ');
            if ~isempty(cf) && isnumeric(cf)
                common_fraction = max(0.0, min(1.0, cf));
            end
        catch
        end
    end

    % Algorithm selection (defaults match Python GUI)
    run_centralized = prompt_bool('Run Centralized? [y/n] (default y): ', true);
    run_fedavg = prompt_bool('Run FedAvg? [y/n] (default y): ', true);
    run_fedprox = prompt_bool('Run FedProx? [y/n] (default n): ', false);
    run_scaffold = prompt_bool('Run SCAFFOLD? [y/n] (default n): ', false);
    run_fedadagrad = prompt_bool('Run FedAdagrad? [y/n] (default n): ', false);
    run_fednova = prompt_bool('Run FedNova? [y/n] (default n): ', false);

    % Compute split sizes (80/10/10)
    train_target = floor(total_dataset * 0.8);
    test_target = floor(total_dataset * 0.1);
    val_target = total_dataset - train_target - test_target;

    if strcmp(dataset_mode, 'non_iid')
        fprintf('Loading Fashion-MNIST (train=%d, val=%d, test=%d) ...\n', train_target, val_target, test_target);
        [train_images, train_labels, val_images, val_labels, test_images, test_labels] = load_fashion_mnist(train_target, val_target, test_target);
        shards = make_non_iid_primary_with_common(train_labels, clients, 'common_label', common_label, 'common_fraction', common_fraction);
    else
        fprintf('Loading MNIST (train=%d, val=%d, test=%d) ...\n', train_target, val_target, test_target);
        [train_images, train_labels, val_images, val_labels, test_images, test_labels] = load_mnist(train_target, val_target, test_target);
        shards = [];
    end

    % Results store (per algorithm). Use sanitized field names (no spaces).
    results_store = struct();

    % Nested function: append metrics into results_store
    function append_metrics(algo_key, metrics)
        if ~isfield(results_store, algo_key)
            s = struct('Accuracy', [], 'Convergence', [], 'CommunicationCost', [], 'StabilityVariance', [], 'TrainingTime', [], 'Velocity', []);
            results_store.(algo_key) = s;
        end
        rs = results_store.(algo_key);
        rs.Accuracy = [rs.Accuracy, metrics.accuracy];
        rs.Convergence = [rs.Convergence, metrics.convergence];
        rs.CommunicationCost = [rs.CommunicationCost, metrics.communication_cost];
        rs.StabilityVariance = [rs.StabilityVariance, metrics.stability_variance];
        rs.TrainingTime = [rs.TrainingTime, metrics.training_time];
        rs.Velocity = [rs.Velocity, metrics.velocity];
        results_store.(algo_key) = rs;
    end

    % Nested function: report log line and store metrics
    function report(algo_name, round_num, metrics)
        fprintf('%s — Round %d: acc=%.4f, conv=%+.4f, comm=%d, var=%.6f, time=%.3fs, vel=%.3f\n', ...
            algo_name, round_num, metrics.accuracy, metrics.convergence, metrics.communication_cost, metrics.stability_variance, metrics.training_time, metrics.velocity);
        append_metrics(lower(algo_name), metrics);
    end

    % Run selected algorithms
    if run_centralized
        fprintf('Running Centralized...\n');
        t0 = tic;
        acc = train_centralized(train_images, train_labels, test_images, test_labels);
        dt = toc(t0);
        m = struct('accuracy', acc, 'convergence', 0.0, 'communication_cost', 0, 'stability_variance', 0.0, 'training_time', dt, 'velocity', 0.0);
        report('Centralized', 1, m);
    end

    if run_fedavg
        fprintf('Running FedAvg (clients=%d, rounds=%d)...\n', clients, rounds);
        fedavg(train_images, train_labels, test_images, test_labels, 'clients', clients, 'rounds', rounds, 'shards', shards, 'report', @report);
    end

    if run_fedprox
        fprintf('Running FedProx...\n');
        fedprox(train_images, train_labels, test_images, test_labels, 'clients', clients, 'rounds', rounds, 'shards', shards, 'report', @report);
    end

    if run_scaffold
        fprintf('Running SCAFFOLD...\n');
        scaffold(train_images, train_labels, test_images, test_labels, 'clients', clients, 'rounds', rounds, 'shards', shards, 'report', @report);
    end

    if run_fedadagrad
        fprintf('Running FedAdagrad...\n');
        fedadagrad(train_images, train_labels, test_images, test_labels, 'clients', clients, 'rounds', rounds, 'shards', shards, 'report', @report);
    end

    if run_fednova
        fprintf('Running FedNova...\n');
        fednova(train_images, train_labels, test_images, test_labels, 'clients', clients, 'rounds', rounds, 'shards', shards, 'report', @report);
    end

    % Write results to src/results as Matlab result (m_result.txt)
    here = fileparts(mfilename('fullpath'));
    results_dir = fullfile(fileparts(here), 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    result_path = fullfile(results_dir, 'm_result.txt');

    fprintf('Saving Matlab results to %s ...\n', result_path);
    fid = fopen(result_path, 'w');
    if fid == -1
        error('Cannot open result.txt for writing');
    end
    algos = fieldnames(results_store);
    for i = 1:numel(algos)
        key = algos{i};
        s = results_store.(key);
        line = sprintf('%s=%s\n', key, python_like_dict(s));
        fprintf(fid, '%s', line);
    end
    fclose(fid);
    fprintf('Done. Results saved.\n');

    % Helper to format lists for Python
    function s = python_like_list(nums)
        if isempty(nums)
            s = '[]'; return;
        end
        if islogical(nums)
            nums = double(nums);
        end
        parts = arrayfun(@(x) sprintf('%.6f', x), nums, 'UniformOutput', false);
        s = ['[' strjoin(parts, ', ') ']'];
    end

    % Helper to format dict for Python with expected key names
    function d = python_like_dict(rs)
        acc = python_like_list(rs.Accuracy);
        conv = python_like_list(rs.Convergence);
        parts = arrayfun(@(x) sprintf('%d', x), rs.CommunicationCost, 'UniformOutput', false);
        comm = sprintf('[%s]', strjoin(parts, ', '));
        varr = python_like_list(rs.StabilityVariance);
        time = python_like_list(rs.TrainingTime);
        vel = python_like_list(rs.Velocity);
        d = sprintf('{''Accuracy'': %s, ''Convergence'': %s, ''Communication Cost'': %s, ''Stability / Variance'': %s, ''Training Time'': %s, ''Velocity'': %s}', acc, conv, comm, varr, time, vel);
    end

    function out = prompt_bool(prompt_text, default_val)
        try
            s = strtrim(lower(input(prompt_text, 's')));
            if isempty(s)
                out = default_val;
            elseif strcmp(s, 'y') || strcmp(s, 'yes') || strcmp(s, '1')
                out = true;
            elseif strcmp(s, 'n') || strcmp(s, 'no') || strcmp(s, '0')
                out = false;
            else
                out = default_val;
            end
        catch
            out = default_val;
        end
    end
end
