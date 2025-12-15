function shards = make_shards(n, clients, full_data_per_client)
% make_shards: Split indices 1..n into `clients` shards.
% If full_data_per_client=true, every client gets the full set 1..n.

    if full_data_per_client
        shards = cell(1, clients);
        for c = 1:clients
            shards{c} = 1:n;
        end
        return;
    end

    idx = randperm(n);
    base = floor(n / clients);
    r = mod(n, clients);
    sizes = repmat(base, 1, clients);
    if r > 0
        sizes(1:r) = sizes(1:r) + 1;
    end

    shards = cell(1, clients);
    startPos = 1;
    for c = 1:clients
        sz = sizes(c);
        if sz > 0
            shards{c} = idx(startPos:startPos+sz-1);
        else
            shards{c} = [];
        end
        startPos = startPos + sz;
    end
end
