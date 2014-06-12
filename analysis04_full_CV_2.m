%% Do a full CV!
% I plan to do a entire cross validation. The goal is do produce a
% reuseable Lasso with CV for simulated data set. 

%% WARNING: This will clear the work space & variables.
clear;clc;
rng(1)  % To make the result replicable

%% You can set the parameters for CV here
% Please the dimension of the data sets
ntrials = 400;
nvoxels = 200;
% Please set the number of folds 
k = 5;
% It needs to know the number of row labels
numRL = 2;
% Some intermedite parameters, for future use
block = ntrials /k /numRL;
size_test = ntrials / k ;

%% Simulate the data
% Add some noise
X = randn(ntrials,nvoxels);   
size(X);
% Add some signals 
X(1:ntrials/numRL,1) = X(1:ntrials/numRL,1) + 1;
X(ntrials/numRL + 1:end,2) = X(ntrials/numRL + 1 : end ,2) + 1;

% Add row label
rowLabels = zeros(ntrials,1);
rowLabels(1:ntrials / numRL ,1) = 1;

% Creates labels for CV 
rowLabelsTrain = zeros(ntrials - size_test,1); 
rowLabelsTrain(1: (ntrials - size_test)/numRL ,1) = 1; 
rowLabelsTest = zeros(size_test,1); 
rowLabelsTest(1:size_test / numRL ,1) = 1; 


%% Create the indices matrix
% This part will outputs indices_test & indices_train, which store the
% indices for CV
ind_A = 1:k;
ind_A = repmat(ind_A', 1, ntrials/k/numRL);
indices = reshape(ind_A',ntrials/numRL, 1);
indices = repmat(indices,numRL,1);

A = 1:ntrials;
A = repmat(A', 1, nvoxels);

for i = 1:k
    % Find the indices for test set & training set
    ind_test = indices == i;
    ind_train = ~ ind_test;

    % Store the indices into a matirx for future use
    indices_test(:,i) = find(ind_test);  
    indices_train(:,i) = find(ind_train); 

end


%% Analysis

% Initialize some variable to store results

acc_test = NaN(100, k);
acc_train = NaN(100, k);


% Start loop. For each CV, do a lasso.
for j = 1:k
    % Subset the data set
    Xtest = X(indices_test(:,j) ,:);
    Xtrain = X(indices_train(:,j) ,:);

    % Fit LASSO
    fit = glmnet(Xtrain, rowLabelsTrain, 'binomial');  

    % Get the number of lambda
    dim_lambda = size(fit.lambda);
    num_lambda = dim_lambda(1);
    
    % Results on training set, which can be ignored 
    predic_train = (Xtrain * fit.beta) + repmat(fit.a0, ntrials - size_test,1) > 0;     % Prediction
    acc_train(:,j) = mean(repmat(rowLabelsTrain,1,num_lambda) == predic_train)';             % Accuracy

    % Results on testing set
    predic_test = (Xtest*fit.beta + repmat(fit.a0, size_test, 1))> 0 ;    % Predictiion
    acc_test(:,j) = mean(repmat(rowLabelsTest,1,num_lambda) == predic_test)';  % Accuracy


end


% Find maximun accuracy for each CV
max_acc = max(acc_test);
disp(['The maximun accuracy(with the best lambda) in each CV are:'])
disp(max_acc')


% For each lambda, find average accuracy
mean_acc_train = mean(acc_train,2); % training set
mean_acc_test = mean(acc_test,2);   % testing set
% Find the maximum accuracy after taking average
max_mean_acc = max(mean_acc_test);
disp(['Average accuracies for each lambda were computed. The maximun accuracy is '])
disp(max_mean_acc)

% Find the indices for the maximun accuracy
find(mean_acc_test == max(mean_acc_test));
% Display how many voxels were used for the best performance
disp(['How many voxels did the Lasso used, to produce the max_accuracy?'])
disp(fit.df(find (mean_acc_test == max(mean_acc_test))))


%% Visualizing the results
% Plot the accracy for the training set, just to compare
plot(mean_acc_train, 'Linewidth', 2)
hold on 
% Plot the accuracy for the testing set
plot(mean_acc_test, 'r', 'Linewidth', 2) 
max_xrange = 0: 0.5: 100;
% Plot the maximun classification accuracy
plot(max_xrange,max_mean_acc, 'g', 'Linewidth', 2)
% Plot the chance line
chance_range = 0: 0.5 : 100;
plot(chance_range, 0.5,'k', 'linewidth', 2)
hold off

axis([1 num_lambda 0 1])
legend('accuary train', 'accuracy test','max - accuracy test',...
    'Location','SouthEast')
% legend( 'accuracy test','max for accuracy test',...
%     'Location','SouthEast')
xlabel('Trails')
ylabel('Accuracy (chance = 0.5)')