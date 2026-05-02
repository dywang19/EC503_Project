%% --- REGULARIZED WEIGHTED LOGISTIC REGRESSION: BCW ---
% 1. Load & Prep
bcw_data = readtable('/Users/dakotawang/Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data', 'FileType', 'text');
bcw_Y = strcmp(table2array(bcw_data(:, 2)), 'M'); 
bcw_X = table2array(bcw_data(:, 3:32));

% 2. Z-Score Normalization & Split
manual_z = @(data) (data - mean(data)) ./ (std(data) + 1e-6);
X_scaled = manual_z(bcw_X);

% Use a fixed seed for reproducibility in your report
rng(42); 
idx = randperm(size(X_scaled,1)); split = floor(0.8 * size(X_scaled,1));
X_train = X_scaled(idx(1:split), :); Y_train = bcw_Y(idx(1:split));
X_test = X_scaled(idx(split+1:end), :); Y_test = bcw_Y(idx(split+1:end));

% 3. Model Parameters (Aligned with SPECT Logic)
[m, n] = size(X_train);
X_train_b = [ones(m, 1) X_train]; 
theta = zeros(n + 1, 1);
lr = 0.1;           % Higher LR for BCW stability
epochs = 3000;      
penalty_weight = 2.0; % BCW is less imbalanced than SPECT
lambda = 0.1;       % L2 Regularization

% 4. Gradient Descent Loop with Weighting & Regularization
for i = 1:epochs
    z = X_train_b * theta;
    h = 1 ./ (1 + exp(-z));
    
    error = h - Y_train;
    weights = ones(m, 1);
    weights(Y_train == 1) = penalty_weight; % Prioritize Malignant misses
    
    % Regularized Gradient Calculation
    reg_term = (lambda / m) * theta;
    reg_term(1) = 0; % Bias term is not regularized
    
    gradient = ((X_train_b' * (weights .* error)) / m) + reg_term;
    theta = theta - lr * gradient;
end

% 5. Test & Evaluation
X_test_b = [ones(size(X_test, 1), 1) X_test];
prob = 1 ./ (1 + exp(-(X_test_b * theta)));
decision_threshold = 0.5; % Standard threshold for BCW
pred = prob >= decision_threshold;

% 6. Detailed Reporting
tp = sum(pred == 1 & Y_test == 1); tn = sum(pred == 0 & Y_test == 0);
fp = sum(pred == 1 & Y_test == 0); fn = sum(pred == 0 & Y_test == 1);

acc = (tp + tn) / length(Y_test);
f1_mal = (2*tp) / (2*tp + fp + fn + 1e-6); 
f1_ben = (2*tn) / (2*tn + fp + fn + 1e-6);
macro_f1 = (f1_mal + f1_ben) / 2;

fprintf('\n--- BCW ENGINEERED LOGISTIC REGRESSION ---\n');
fprintf('Accuracy: %.3f | Macro-F1: %.3f\n', acc, macro_f1);
fprintf('F1-Malignant: %.3f | F1-Benign: %.3f\n', f1_mal, f1_ben);
fprintf('False Negatives: %d | False Positives: %d\n', fn, fp);
