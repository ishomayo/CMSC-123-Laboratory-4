clear all; close all; clc;

disp('Grey Wolf Optimization for Neural Network Training');
disp('================================================');

% Load and preprocess the bank.csv dataset
disp('Step 1: Loading and preprocessing data...');
[X, y, X_norm, feature_names] = preprocessBankData('bank.csv');

% Display some information about the data
disp(['Number of examples: ' num2str(size(X_norm, 1))]);
disp(['Number of features: ' num2str(size(X_norm, 2))]);
disp(['Number of classes: ' num2str(length(unique(y)))]);

% Split data into train and test sets (70-30 split)
disp('Step 2: Splitting data into training and testing sets...');
rand_idx = randperm(size(X_norm, 1));
X_norm = X_norm(rand_idx, :);
y = y(rand_idx);

train_size = floor(0.7 * size(X_norm, 1));
X_train = X_norm(1:train_size, :);
y_train = y(1:train_size);
X_test = X_norm(train_size+1:end, :);
y_test = y(train_size+1:end);

% Set up neural network parameters
input_layer_size = size(X_train, 2);
hidden_layer_size = 25;
num_labels = length(unique(y_train));

% Set up GWO parameters
max_iter = 30;           % Maximum iterations
search_agents_no = 8;    % Number of search agents (wolves)
lb = -1;                 % Lower bound of weights
ub = 1;                  % Upper bound of weights
lambda = 0.1;            % Regularization parameter

% Calculate dimension (total number of weights)
dim = hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1);

disp('Step 3: Running Grey Wolf Optimization...');
disp(['Neural Network structure: ' num2str(input_layer_size) ' input -> ' ...
      num2str(hidden_layer_size) ' hidden -> ' num2str(num_labels) ' output']);
disp(['GWO parameters: ' num2str(max_iter) ' iterations, ' ...
      num2str(search_agents_no) ' search agents']);

% Run Grey Wolf Optimization
[Alpha_pos, Alpha_score, Convergence_curve] = greyWolfOptimizer(X_train, y_train, ...
    input_layer_size, hidden_layer_size, num_labels, max_iter, search_agents_no, lb, ub, dim);

% Reshape the best weights to Theta1 and Theta2
Theta1 = reshape(Alpha_pos(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(Alpha_pos((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Evaluate the trained model
disp('Step 4: Evaluating model performance...');

% Calculate training set accuracy
pred_train = predict(Theta1, Theta2, X_train);
train_accuracy = mean(double(pred_train == y_train)) * 100;
fprintf('Training Set Accuracy: %f%%\n', train_accuracy);

% Calculate test set accuracy
pred_test = predict(Theta1, Theta2, X_test);
test_accuracy = mean(double(pred_test == y_test)) * 100;
fprintf('Test Set Accuracy: %f%%\n', test_accuracy);

% Plot the convergence curve
disp('Step 5: Plotting convergence curve...');
figure;
plot(1:max_iter, Convergence_curve, 'LineWidth', 2);
title('Convergence Curve');
xlabel('Iteration');
ylabel('Cost Function Value');
grid on;

disp('Optimization completed successfully!');
