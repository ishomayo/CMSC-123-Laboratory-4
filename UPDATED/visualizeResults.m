function visualizeResults(X, y, Theta1, Theta2, convergence_curve)
% Visualize results of the GWO-optimized neural network
% Octave compatible version

% Create figure for convergence curve
figure;
plot(convergence_curve, 'LineWidth', 2);
title('Convergence Curve');
xlabel('Iteration');
ylabel('Cost');
grid on;

% Create confusion matrix visualization
figure;
pred = predict(Theta1, Theta2, X);

% Create confusion matrix manually (Octave doesn't have confusionmat)
classes = unique(y);
n_classes = length(classes);
confMatrix = zeros(n_classes, n_classes);

for i = 1:length(y)
    true_idx = find(classes == y(i));
    pred_idx = find(classes == pred(i));
    confMatrix(true_idx, pred_idx) = confMatrix(true_idx, pred_idx) + 1;
end

% Calculate metrics
accuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
if size(confMatrix, 1) == 2
    precision = confMatrix(2,2) / sum(confMatrix(:,2));
    recall = confMatrix(2,2) / sum(confMatrix(2,:));
    f1 = 2 * precision * recall / (precision + recall);

    % Display metrics
    subplot(1,2,1);
    imagesc(confMatrix);
    colorbar;
    title('Confusion Matrix');
    xlabel('Predicted Class');
    ylabel('True Class');

    subplot(1,2,2);
    bar([accuracy, precision, recall, f1]);
    title('Performance Metrics');
    xticklabels({'Accuracy', 'Precision', 'Recall', 'F1-Score'});
    ylim([0 1]);
    grid on;

    % Print results
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall: %.4f\n', recall);
    fprintf('F1-Score: %.4f\n', f1);
else
    % For multi-class problems
    imagesc(confMatrix);
    colorbar;
    title('Confusion Matrix');
    xlabel('Predicted Class');
    ylabel('True Class');

    fprintf('Accuracy: %.4f\n', accuracy);
end

% Plot feature importance if applicable
if size(Theta1, 2) <= 20
    figure;
    feature_importance = sum(abs(Theta1(:, 2:end)), 1);
    bar(feature_importance);
    title('Feature Importance');
    xlabel('Feature Index');
    ylabel('Importance (Sum of Absolute Weights)');
    grid on;
end

end

function performSensitivityAnalysis(X, y, input_layer_size, hidden_layer_size, num_labels)
% Perform sensitivity analysis on GWO parameters

% Parameters to analyze
max_iters = [30, 50, 100];
search_agents = [5, 10, 20];
lambdas = [0.01, 0.1, 0.5];
hidden_neurons = [10, 25, 50];

% Prepare results storage
results = zeros(length(max_iters), length(search_agents), length(lambdas), length(hidden_neurons));

% Function to evaluate network with specific parameters
function acc = evaluateNetwork(max_iter, agents, lambda, hidden)
    dim = hidden * (input_layer_size + 1) + num_labels * (hidden + 1);
    [best_weights, ~, ~] = greyWolfOptimizer(X, y, input_layer_size, hidden, num_labels, max_iter, agents, -1, 1, dim, lambda);

    % Reshape weights
    Theta1 = reshape(best_weights(1:hidden * (input_layer_size + 1)), hidden, (input_layer_size + 1));
    Theta2 = reshape(best_weights((1 + (hidden * (input_layer_size + 1))):end), num_labels, (hidden + 1));

    % Calculate accuracy
    pred = predict(Theta1, Theta2, X);
    acc = mean(double(pred == y)) * 100;
end

% Run parameter grid search
for i = 1:length(max_iters)
    for j = 1:length(search_agents)
        for k = 1:length(lambdas)
            for l = 1:length(hidden_neurons)
                fprintf('Testing: max_iter=%d, agents=%d, lambda=%.2f, hidden=%d\n', ...
                       max_iters(i), search_agents(j), lambdas(k), hidden_neurons(l));

                results(i,j,k,l) = evaluateNetwork(max_iters(i), search_agents(j), lambdas(k), hidden_neurons(l));
            end
        end
    end
end

% Visualize results
figure;
subplot(2,2,1);
imagesc(squeeze(mean(mean(results, 3), 4)));
title('Effect of Iterations and Agents');
xlabel('Number of Agents');
ylabel('Max Iterations');
xticklabels(search_agents);
yticklabels(max_iters);
colorbar;

subplot(2,2,2);
imagesc(squeeze(mean(mean(results, 2), 4)));
title('Effect of Iterations and Lambda');
xlabel('Lambda');
ylabel('Max Iterations');
xticklabels(lambdas);
yticklabels(max_iters);
colorbar;

subplot(2,2,3);
imagesc(squeeze(mean(mean(results, 2), 3)));
title('Effect of Iterations and Hidden Neurons');
xlabel('Hidden Neurons');
ylabel('Max Iterations');
xticklabels(hidden_neurons);
yticklabels(max_iters);
colorbar;

subplot(2,2,4);
imagesc(squeeze(mean(mean(results, 1), 3)));
title('Effect of Agents and Hidden Neurons');
xlabel('Hidden Neurons');
ylabel('Number of Agents');
xticklabels(hidden_neurons);
yticklabels(search_agents);
colorbar;

sgtitle('Sensitivity Analysis of GWO Parameters');

% Print best configuration
[max_val, idx] = max(results(:));
[i, j, k, l] = ind2sub(size(results), idx);
fprintf('\nBest configuration: Accuracy = %.2f%%\n', max_val);
fprintf('max_iter = %d\n', max_iters(i));
fprintf('agents = %d\n', search_agents(j));
fprintf('lambda = %.2f\n', lambdas(k));
fprintf('hidden neurons = %d\n', hidden_neurons(l));

end
