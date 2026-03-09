function [train_images, train_labels, val_images, val_labels, test_images, test_labels] = load_fashion_mnist(train_size, val_size, test_size)
% load_fashion_mnist: Download and load Fashion-MNIST (IDX format).
% Returns double arrays normalized to [0,1]. Shapes: N x 28 x 28, labels N x 1.

    if nargin < 1, train_size = 3000; end
    if nargin < 2, val_size = 0; end
    if nargin < 3, test_size = 1000; end

    here = fileparts(mfilename('fullpath'));
    dataDir = fullfile(here, 'data');
    if ~exist(dataDir, 'dir'), mkdir(dataDir); end

    files = struct();
    files.trainImagesGz = fullfile(dataDir, 'fashion-train-images-idx3-ubyte.gz');
    files.trainLabelsGz = fullfile(dataDir, 'fashion-train-labels-idx1-ubyte.gz');
    files.testImagesGz  = fullfile(dataDir, 'fashion-t10k-images-idx3-ubyte.gz');
    files.testLabelsGz  = fullfile(dataDir, 'fashion-t10k-labels-idx1-ubyte.gz');

    % Fashion-MNIST source files from official repository.
    base = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/';
    urls = struct();
    urls.trainImagesGz = [base 'train-images-idx3-ubyte.gz'];
    urls.trainLabelsGz = [base 'train-labels-idx1-ubyte.gz'];
    urls.testImagesGz  = [base 't10k-images-idx3-ubyte.gz'];
    urls.testLabelsGz  = [base 't10k-labels-idx1-ubyte.gz'];

    files = ensure_download(files, urls);

    trainImagesFile = gunzip_if_needed(files.trainImagesGz, dataDir);
    trainLabelsFile = gunzip_if_needed(files.trainLabelsGz, dataDir);
    testImagesFile  = gunzip_if_needed(files.testImagesGz, dataDir);
    testLabelsFile  = gunzip_if_needed(files.testLabelsGz, dataDir);

    train_images_all = read_idx_images(trainImagesFile);
    train_labels_all = read_idx_labels(trainLabelsFile);
    test_images_all  = read_idx_images(testImagesFile);
    test_labels_all  = read_idx_labels(testLabelsFile);

    train_images_all = double(train_images_all) / 255.0;
    test_images_all  = double(test_images_all) / 255.0;

    total_train = size(train_images_all, 1);
    t_size = max(0, min(train_size, total_train));
    v_size = max(0, min(val_size, max(0, total_train - t_size)));
    te_size = max(0, min(test_size, size(test_images_all, 1)));

    train_images = train_images_all(1:t_size, :, :);
    train_labels = train_labels_all(1:t_size);
    val_images = train_images_all(t_size+1:t_size+v_size, :, :);
    val_labels = train_labels_all(t_size+1:t_size+v_size);

    test_images = test_images_all(1:te_size, :, :);
    test_labels = test_labels_all(1:te_size);
end

function files = ensure_download(files, urls)
    fns = fieldnames(files);
    for i = 1:numel(fns)
        fp = files.(fns{i});
        if ~exist(fp, 'file')
            websave(fp, urls.(fns{i}));
        end
    end
end

function outFile = gunzip_if_needed(gzFile, outDir)
    [~, base, ext] = fileparts(gzFile);
    if strcmp(ext, '.gz')
        outFile = fullfile(outDir, base);
        if ~exist(outFile, 'file')
            gunzip(gzFile, outDir);
        end
    else
        outFile = gzFile;
    end
end

function X = read_idx_images(filename)
    fid = fopen(filename, 'rb');
    assert(fid ~= -1, 'Cannot open %s', filename);
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, 'Invalid magic number for images');
    num = fread(fid, 1, 'int32', 0, 'ieee-be');
    rows = fread(fid, 1, 'int32', 0, 'ieee-be');
    cols = fread(fid, 1, 'int32', 0, 'ieee-be');
    data = fread(fid, num * rows * cols, 'uint8');
    fclose(fid);
    X = reshape(data, [rows, cols, num]);
    X = permute(X, [3 1 2]);
end

function y = read_idx_labels(filename)
    fid = fopen(filename, 'rb');
    assert(fid ~= -1, 'Cannot open %s', filename);
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, 'Invalid magic number for labels');
    num = fread(fid, 1, 'int32', 0, 'ieee-be');
    data = fread(fid, num, 'uint8');
    fclose(fid);
    y = double(data);
end
