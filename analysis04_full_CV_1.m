%% Do a full CV!
clear;clc;

%% You can set the parameters for CV here
% Please the dimension of the data sets
ntrials = 200;
nvoxels = 100;
% Please set the number of folds 
k = 5;
% It needs to know the number of row labels
numRL = 2;

block = ntrials /k /numRL;
size_test = ntrials / k 

%% Simulate data
% Add some noise
X = randn(ntrials,nvoxels);   
size(X);
% Add some signals (suppose there are 2 row labels )
X(1:ntrials/2,1) = X(1:ntrials/2,1) + 1;
X(ntrials/2 + 1:end,2) = X(ntrials/2 + 1 : end ,2) + 1;

% Add row label
rowLabels = zeros(ntrials,1);
rowLabels(1:ntrials /2 ,1) = 1;

% Creates labels for CV 
rowLabelsTrain = zeros(ntrials - size_test,1); 
rowLabelsTrain(1: (ntrials - size_test)/2 ,1) = 1; 
rowLabelsTest = zeros(size_test,1); 
rowLabelsTest(1:size_test / 2 ,1) = 1; 


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

    % This part can check whether the indices assignment is correct
    % It could be ignored/commented
    ind_test = find(ind_test);
    ind_train = find(ind_train);
    disp(['This is the ' num2str(i) 'th CV, the test set indices are: '])
    disp(ind_test')  

end


%% Subset the data set
Xtest = X(indices_test(:,1) ,:);
Xtrain = X(indices_train(:,1) ,:);

%% Analysis
% Fit LASSO
fit = glmnet(Xtrain, rowLabelsTrain, 'binomial');  
dim_lambda = size(fit.lambda);
num_lambda = dim_lambda(1);

% Results on training set, which can be ignored
(Xtrain * fit.beta) + repmat(fit.a0, ntrials - size_test,1);   
predic_train = (Xtrain * fit.beta) + repmat(fit.a0, ntrials - size_test(1),1) > 0;   % prediction
repmat(rowLabelsTrain,1,num_lambda) == predic_train;          % compare prediction with truth
acc_train = mean(repmat(rowLabelsTrain,1,num_lambda) == predic_train)';   % accuracy

% Results on testing set
Xtest*fit.beta + repmat(fit.a0, size_test(1), 1);          % prediction
predic_test = (Xtest*fit.beta + repmat(fit.a0, size_test(1), 1))> 0 ;    % How well does it fits the truth
predic_test == repmat(rowLabelsTest,1,num_lambda);
acc_test = mean(repmat(rowLabelsTest,1,num_lambda) == predic_test)' 

max_acc = max(acc_test)
max_ind = find(acc_test == max(acc_test)); % store the indices for the best lambdas


%% Visualizing the results
% Plot the accracy for the training set, just to compare
plot(acc_train, 'Linewidth', 2)
hold on 
% Plot the accuracy for the testing set
plot(acc_test, 'r', 'Linewidth', 2) 
max_xrange = 0: 0.5: 100;
% Plot the maximun classification accuracy
plot(max_xrange,max_acc, 'g', 'Linewidth', 2)
% Plot the chance line
chance_range = 0: 0.5 : 100;
plot(chance_range, 0.5,'k', 'linewidth', 2)
hold off

axis([1 num_lambda 0 1])
legend('accuary train', 'accuracy test','max for accuracy test',...
    'Location','SouthEast')
xlabel('Trails')
ylabel('Accuracy (chance = 0.5)')