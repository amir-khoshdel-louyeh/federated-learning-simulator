function [train_images, train_labels, val_images, val_labels, test_images, test_labels] = load_mnist(train_size, val_size, test_size)
% load_mnist: Download and load MNIST dataset (IDX format) into MATLAB arrays.
% Returns double arrays normalized to [0,1]. Shapes: N x 28 x 28, labels N x 1.
% If files already exist under Matlab/data/, reuse them.

    if nargin < 1, train_size = 3000; end
    if nargin < 2, val_size = 0; end
    if nargin < 3, test_size = 1000; end

    here = fileparts(mfilename('fullpath'));
    dataDir = fullfile(here, 'data');
    if ~exist(dataDir, 'dir'), mkdir(dataDir); end

    files = struct();
    files.trainImagesGz = fullfile(dataDir, 'train-images-idx3-ubyte.gz');
    files.trainLabelsGz = fullfile(dataDir, 'train-labels-idx1-ubyte.gz');
    files.testImagesGz  = fullfile(dataDir, 't10k-images-idx3-ubyte.gz');
    files.testLabelsGz  = fullfile(dataDir, 't10k-labels-idx1-ubyte.gz');

    urls = struct();
    base = 'http://yann.lecun.com/exdb/mnist/';
    urls.trainImagesGz = [base 'train-images-idx3-ubyte.gz'];
    urls.trainLabelsGz = [base 'train-labels-idx1-ubyte.gz'];
    urls.testImagesGz  = [base 't10k-images-idx3-ubyte.gz'];
    urls.testLabelsGz  = [base 't10k-labels-idx1-ubyte.gz'];

    % Download if missing
    files = ensure_download(files, urls);

    % Unzip if needed
    trainImagesFile = gunzip_if_needed(files.trainImagesGz, dataDir);
    trainLabelsFile = gunzip_if_needed(files.trainLabelsGz, dataDir);
    testImagesFile  = gunzip_if_needed(files.testImagesGz, dataDir);
    testLabelsFile  = gunzip_if_needed(files.testLabelsGz, dataDir);

    % Read IDX
    train_images_all = read_idx_images(trainImagesFile);
    train_labels_all = read_idx_labels(trainLabelsFile);
    test_images_all  = read_idx_images(testImagesFile);
    test_labels_all  = read_idx_labels(testLabelsFile);

    % Normalize
    train_images_all = double(train_images_all) / 255.0;
    test_images_all  = double(test_images_all) / 255.0;

    total_train = size(train_images_all, 1);
    t_size = max(0, min(train_size, total_train));
    v_size = max(0, min(val_size, max(0, total_train - t_size)));
    te_size = max(0, min(test_size, size(test_images_all, 1)));

    t_imgs = train_images_all(1:t_size, :, :);
    t_lbls = train_labels_all(1:t_size);
    v_imgs = train_images_all(t_size+1:t_size+v_size, :, :);
    v_lbls = train_labels_all(t_size+1:t_size+v_size);

    te_imgs = test_images_all(1:te_size, :, :);
    te_lbls = test_labels_all(1:te_size);

    train_images = t_imgs; train_labels = t_lbls;
    val_images = v_imgs;   val_labels   = v_lbls;
    test_images = te_imgs; test_labels  = te_lbls;
end

function files = ensure_download(files, urls)
    fns = fieldnames(files);
    for i = 1:numel(fns)
        fp = files.(fns{i});
        if ~exist(fp, 'file')
            try
                websave(fp, urls.(fns{i}));
            catch
                % Retry with https if http blocked
                baseHTTPS = 'https://ossci-datasets.s3.amazonaws.com/mnist/';
                alt = struct();
                alt.trainImagesGz = [baseHTTPS 'train-images-idx3-ubyte.gz'];
                alt.trainLabelsGz = [baseHTTPS 'train-labels-idx1-ubyte.gz'];
                alt.testImagesGz  = [baseHTTPS 't10k-images-idx3-ubyte.gz'];
                alt.testLabelsGz  = [baseHTTPS 't10k-labels-idx1-ubyte.gz'];
                websave(fp, alt.(fns{i}));
            end
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
    data = fread(fid, num*rows*cols, 'uint8');
    fclose(fid);
    X = reshape(data, [rows, cols, num]);
    X = permute(X, [3 1 2]); % N x rows x cols
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
