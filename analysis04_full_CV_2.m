%% Do a full CV!
% I plan to do a entire cross validation. The goal is to build a
% reuseable Lasso with CV for simulated data set. 

%% WARNING: This will clear the work space & variables.
clear;clc;
rng(1)  % To make the result replicable

%% You can set the parameters for CV here
% Please the dimension of the data sets
ntrials = 150;
nvoxels = 150;
% Please set the number of folds 
k = 5;
% It needs to know the number of row labels
% ps:Currently, this program can just run normally when there are 2 rowlabels
rowLabels.num = 2;
% Some intermedite parameters, for future use
block = ntrials /k /rowLabels.num;
test.size = ntrials / k ;


%% Simulate the data

% Creating the background
X = zeros(ntrials, nvoxels);

% Add some signals 
signal = .5;

X(1:ntrials/rowLabels.num,1) = X(1:ntrials/rowLabels.num,1) + signal;
X(ntrials/rowLabels.num + 1:end,2) = X(ntrials/rowLabels.num + 1 : end ,2) + signal;

X(1:ntrials/rowLabels.num,3) = X(1:ntrials/rowLabels.num,3) + signal;
X(ntrials/rowLabels.num + 1:end,4) = X(ntrials/rowLabels.num + 1 : end ,4) + signal;

X(1:ntrials/rowLabels.num,5) = X(1:ntrials/rowLabels.num,5) + signal;
X(ntrials/rowLabels.num + 1:end,6) = X(ntrials/rowLabels.num + signal : end ,6) + signal;


% plot
figure(1)
imagesc(X)

% Adding noise 
rng(1) % Set the seeTo make the result replicable
X = X + randn(ntrials,nvoxels);   
size(X);

figure(2)
imagesc(X)

% Add row label
rowLabels.whole = zeros(ntrials,1);
rowLabels.whole(1:ntrials / rowLabels.num ,1) = 1;

% Creates labels for CV 
rowLabels.train = zeros(ntrials - test.size,1); 
rowLabels.train(1: (ntrials - test.size)/rowLabels.num ,1) = 1; 
rowLabels.test = zeros(test.size,1); 
rowLabels.test(1:test.size / rowLabels.num ,1) = 1; 


%% Create the indices matrix
% This part will outputs test.indices & train.indices, which store the
% indices for CV
CV.indices = 1:k;
CV.indices = repmat(CV.indices', 1, ntrials/k/rowLabels.num);
CV.indices = reshape(CV.indices',ntrials/rowLabels.num, 1);
CV.indices = repmat(CV.indices,rowLabels.num,1);

for i = 1:k
    % Find the indices for test set & training set
    test.ind = CV.indices == i;
    train.ind = ~ test.ind;

    % Store the indices into a matirx for future use
    test.indices(:,i) = find(test.ind);  
    train.indices(:,i) = find(train.ind); 

end


%% Fit Lasso

% Initialize some variable to store results
test.accuracy = NaN(100, k);
train.accuracy = NaN(100, k);

% Start loop. For each CV, do a lasso.
for i = 1:k
    % Subset the data set
    Xtest = X(test.indices(:,i) ,:);
    Xtrain = X(train.indices(:,i) ,:);

    % Fit LASSO
    fit = glmnet(Xtrain, rowLabels.train, 'binomial');  

    % Get the number of lambda
    lambda.dim = size(fit.lambda);
    lambda.num = lambda.dim(1);
    
    % Results on training set, which can be ignored 
    train.prediction = (Xtrain * fit.beta) + repmat(fit.a0, ntrials - test.size,1) > 0;   
    % Store the results
    train.accuracy(:,i) = mean(repmat(rowLabels.train,1,lambda.dim) == train.prediction)';          

    % Results on testing set
    test.prediction = (Xtest*fit.beta + repmat(fit.a0, test.size, 1))> 0 ;    
    % Store the results
    test.accuracy(:,i) = mean(repmat(rowLabels.test,1,lambda.dim) == test.prediction)';  

end

%% Analysis

% Find maximun accuracy for each CV
max_acc = max(test.accuracy);
disp(['The maximun accuracy(with the best lambda) in each CV are:'])
disp(max_acc')

% For each lambda, find average accuracy
train.meanAccuracy = mean(train.accuracy,2); % training set
test.meanAccuracy = mean(test.accuracy,2);   % testing set
% Find the maximum accuracy after taking average
max_mean_acc = max(test.meanAccuracy);
disp(['Average accuracies for each lambda were computed. The maximun accuracy is '])
disp(max_mean_acc)

% Find the indices for the maximun accuracy
lambda.IndBest =find(test.meanAccuracy == max(test.meanAccuracy));
% The best lambda values
fit.lambda(find(test.meanAccuracy == max(test.meanAccuracy)));

% Display how many voxels were used for the best performance
disp(['How many voxels did the Lasso used, to produce the max_accuracy?'])
disp(fit.df(find (test.meanAccuracy == max(test.meanAccuracy))))



%% Preparation for Iterative Lasso

% Show me the nonzero beta(voxels that have been used) for the best lambda 
beta.value = fit.beta(:,lambda.IndBest);
beta.size = size(beta.value);

% for i = 1:beta.size(2)
%     tabulate(beta.value(:,i));
% end


% Give me the indices for these nonzero beta
for i = 1: beta.size(2)
    
    find( beta.value(:,i) ~= 0 )
end





%% Visualizing the results

% Weights Plot
figure(3)
imagesc(fit.beta)
xlabel('Lambda')
ylabel('Voxels')
title ('Weights plot for every voxels ')

figure(4)
glmnetPlot(fit)

% Accuracy Plot
figure(5)
% Plot the accracy for the training set, just to compare
plot(train.meanAccuracy, 'Linewidth', 2)
hold on 
% Plot the accuracy for the testing set
plot(test.meanAccuracy, 'r', 'Linewidth', 2) 
max_xrange = 0: 0.5: 100;
% Plot the maximun classification accuracy
plot(max_xrange,max_mean_acc, 'g', 'Linewidth', 2)
% Plot the chance line
chance.rate = 1 / rowLabels.num;
chance.range = 0: 0.5 : 100;
plot(chance.range, chance.rate ,'k', 'linewidth', 2)
hold off

title('Accuracy Plot for every lambda values')
axis([1 lambda.num 0 1])
xlabel('Lambda')
ylabel('Accuracy (chance = 0.5)')
legend('accuary train', 'accuracy test','max - accuracy test',...
    'Location','SouthEast')
