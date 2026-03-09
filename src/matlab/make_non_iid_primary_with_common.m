function shards = make_non_iid_primary_with_common(y_train, clients, varargin)
% Create non-IID shards: each client has one primary label and a shared
% portion of one common label (matching Python partition behavior).
%
% Name-value options:
%  'primary_labels' : vector of length clients (default auto)
%  'common_label'   : scalar label shared across clients (default 7)
%  'common_fraction': fraction of common_label samples to distribute (default 0.1)

    p = inputParser;
    addParameter(p, 'primary_labels', []);
    addParameter(p, 'common_label', 7);
    addParameter(p, 'common_fraction', 0.1);
    parse(p, varargin{:});

    primary_labels = p.Results.primary_labels;
    common_label = p.Results.common_label;
    common_fraction = p.Results.common_fraction;

    common_fraction = max(0.0, min(1.0, common_fraction));
    labels = unique(y_train(:))';
    labels_no_common = labels(labels ~= common_label);

    if isempty(primary_labels)
        if isempty(labels_no_common)
            error('No available primary labels in y_train');
        end
        % Mirror Python behavior: shuffled unique labels and cycle if needed.
        labels_no_common = labels_no_common(randperm(numel(labels_no_common)));
        primary_labels = zeros(1, clients);
        for c = 1:clients
            idx = mod(c - 1, numel(labels_no_common)) + 1;
            primary_labels(c) = labels_no_common(idx);
        end
    else
        if numel(primary_labels) ~= clients
            error('primary_labels must have length equal to clients');
        end
        primary_labels = reshape(primary_labels, 1, []);
    end

    shards = cell(1, clients);
    for c = 1:clients
        shards{c} = [];
    end

    % Distribute shared common-label portion equally among clients.
    common_idx = find(y_train == common_label);
    common_idx = common_idx(randperm(numel(common_idx)));
    total_common = floor(numel(common_idx) * common_fraction);
    common_selected = common_idx(1:total_common);
    parts = split_indices(common_selected, clients);
    for c = 1:clients
        shards{c} = [shards{c}, parts{c}]; %#ok<AGROW>
    end

    % Add all samples for each client's primary label.
    for c = 1:clients
        p_label = primary_labels(c);
        p_idx = find(y_train == p_label);
        p_idx = p_idx(randperm(numel(p_idx)));
        shards{c} = [shards{c}, p_idx(:)']; %#ok<AGROW>
    end

    % Final per-client shuffle.
    for c = 1:clients
        s = shards{c};
        if ~isempty(s)
            shards{c} = s(randperm(numel(s)));
        end
    end
end

function parts = split_indices(idx, k)
    n = numel(idx);
    base = floor(n / k);
    r = mod(n, k);
    sizes = repmat(base, 1, k);
    if r > 0
        sizes(1:r) = sizes(1:r) + 1;
    end
    parts = cell(1, k);
    pos = 1;
    for i = 1:k
        sz = sizes(i);
        if sz > 0
            parts{i} = idx(pos:pos+sz-1);
        else
            parts{i} = [];
        end
        pos = pos + sz;
    end
end
