% Evaluating AdaBoost on SPECT
spect_train = readmatrix('SPECT.train', 'FileType', 'text');
X_train = spect_train(:,2:end);
Y_train = spect_train(:,1);

spect_test = readmatrix('SPECT.test', 'FileType', 'text');
X_test = spect_test(:,2:end);
Y_test = spect_test(:,1);

iterations = 2;

model = imbalancedAdaBoost(X_train, Y_train, iterations);