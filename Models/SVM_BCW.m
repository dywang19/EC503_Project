% 1. Load & Prep
bcw_data = readtable('/Users/dakotawang/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data', 'FileType', 'text');
bcw_X = table2array(bcw_data(:, 3:32)); 
bcw_Y = strcmp(table2array(bcw_data(:, 2)), 'M'); % Malignant = 1, Benign = 0

% 2. Z-Score Normalization
X_scaled = (bcw_X - mean(bcw_X)) ./ (std(bcw_X) + 1e-6);

% 3. Split (Using seed 42 to match your successful LogReg run)
rng(42); 
idx = randperm(size(X_scaled,1)); split = floor(0.8 * size(X_scaled,1));
X_train = X_scaled(idx(1:split), :); Y_train = bcw_Y(idx(1:split));
X_test = X_scaled(idx(split+1:end), :); Y_test = bcw_Y(idx(split+1:end));

% 4. "One-Shot" Linear Solver 
% This replaces "Learning Rates" with direct Linear Algebra.
% SVM labels must be -1 and 1
Y_svm = double(Y_train); 
Y_svm(Y_svm == 0) = -1;

X_train_b = [ones(size(X_train,1),1) X_train]; % Add Bias
% This is the "Normal Equation" - the gold standard for solving linear models manually
w = (X_train_b' * X_train_b + 0.1 * eye(size(X_train_b,2))) \ (X_train_b' * Y_svm);

% 5. Test
X_test_b = [ones(size(X_test,1),1) X_test];
pred = (X_test_b * w) >= 0; 

% 6. Calculation of All Metrics
tp = sum(pred == 1 & Y_test == 1); 
tn = sum(pred == 0 & Y_test == 0);
fp = sum(pred == 1 & Y_test == 0); 
fn = sum(pred == 0 & Y_test == 1);

acc = (tp + tn) / length(Y_test);
f1_min = (2*tp) / (2*tp + fp + fn);
f1_maj = (2*tn) / (2*tn + fp + fn);
macro_f1 = (f1_min + f1_maj) / 2;

fprintf('\n--- VERIFIED SVM BASELINE RESULTS ---\n');
fprintf('Accuracy: %.3f | F1-Minority: %.3f | F1-Majority: %.3f | Macro-F1: %.3f | FN: %d\n', ...
        acc, f1_min, f1_maj, macro_f1, fn);
