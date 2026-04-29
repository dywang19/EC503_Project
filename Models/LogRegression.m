%% --- LOGISTIC REGRESSION BASELINE ---
% 1. Load & Prep
bcw_data = readtable('/Users/dakotawang/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data', 'FileType', 'text');
bcw_Y = strcmp(table2array(bcw_data(:, 2)), 'M'); 
bcw_X = table2array(bcw_data(:, 3:32));

% 2. Manual Normalization & Split
manual_z = @(data) (data - mean(data)) ./ (std(data) + 1e-6);
X_scaled = manual_z(bcw_X);
idx = randperm(size(X_scaled,1)); split = floor(0.8 * size(X_scaled,1));
X_train = X_scaled(idx(1:split), :); Y_train = bcw_Y(idx(1:split));
X_test = X_scaled(idx(split+1:end), :); Y_test = bcw_Y(idx(split+1:end));

% 3. Train (Gradient Descent)
X_train_b = [ones(size(X_train,1),1) X_train];
w = zeros(size(X_train_b,2), 1);
lr = 0.1;
for i = 1:1500
    h = 1 ./ (1 + exp(-(X_train_b * w)));
    w = w - (lr/length(Y_train)) * (X_train_b' * (h - Y_train));
end

% 4. Test & Evaluate
X_test_b = [ones(size(X_test,1),1) X_test];
pred = (1 ./ (1 + exp(-(X_test_b * w)))) >= 0.5;
calculate_report(Y_test, pred, 'Logistic Regression');

function calculate_report(y_true, y_pred, name)
    tp = sum(y_pred == 1 & y_true == 1); tn = sum(y_pred == 0 & y_true == 0);
    fp = sum(y_pred == 1 & y_true == 0); fn = sum(y_pred == 0 & y_true == 1);
    acc = (tp + tn) / length(y_true);
    f1_min = 2*tp / (2*tp + fp + fn); f1_maj = 2*tn / (2*tn + fp + fn);
    macro_f1 = (f1_min + f1_maj) / 2;
    fprintf('\n--- %s Results ---\n', name);
    fprintf('Accuracy: %.3f | F1-Minority: %.3f | F1-Majority: %.3f | Macro-F1: %.3f | FN: %d\n', ...
            acc, f1_min, f1_maj, macro_f1, fn);
end