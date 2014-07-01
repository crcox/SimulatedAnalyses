%% Iterative Lasso 

%% WARNING: This will clear the work space & variables.
clear;clc;

%% You can set the parameters for CV here
% Please the dimension of the data sets 
ntrials = 100; % it has to be divisible by K
nvoxels = 100;
% Please set the number of folds 
k = 5;
% It needs to know the number of row labels
% ps:Currently, this program can just run normally when there are 2 rowlabels
rowLabels.num = 2;
% It is useful to know the size for the testing set
test.size = ntrials / k ;

% Display the parameters
disp ('Parameters: ')
disp(['Voxels: '  num2str(nvoxels)])
disp(['trials: '  num2str(ntrials)])
disp(['K : '  num2str(k)])
disp(['Number of row labels: ' num2str(rowLabels.num)])



%% Simulate the data
% Creating the background 
X.raw = zeros(ntrials, nvoxels);

% % Add some signals 
signal = 1;
disp(['Signal intensity = ' num2str(signal)])
X.raw(1:ntrials/rowLabels.num,1:15) = X.raw(1:ntrials/rowLabels.num,1:15) + signal;
X.raw(ntrials/rowLabels.num + 1:end,16:30) = X.raw(ntrials/rowLabels.num + 1 : end ,16:30) + signal;
% plot the signal
figure(1)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal');

% Adding noise 
rng(1) % To make the result replicable
noise = 1;
disp(['Noise intensity = ' num2str(noise)])
X.raw = X.raw + noise * randn(ntrials,nvoxels);   
size(X.raw);
% plot the noise + signal
figure(2)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal & Noise')

% Create row labels
rowLabels.whole = zeros(ntrials,1);
rowLabels.whole(1:ntrials / rowLabels.num ,1) = 1;
% Create new row labels for CV blocks
rowLabels.train = zeros(ntrials - test.size,1); 
rowLabels.train(1: (ntrials - test.size)/rowLabels.num ,1) = 1; 
rowLabels.test = zeros(test.size,1); 
rowLabels.test(1:test.size / rowLabels.num ,1) = 1; 



%% Generating indices for Outer CV
% generate indices for CV
CV.blocks = 1:k;
CV.blocks = CV.blocks';
CV.indices = repmat(CV.blocks,[ntrials / k,1]);
% Generate indices for CVglmnet
CV2.blocks = 1:k-1;
CV2.blocks = CV2.blocks';
CV2.indices = repmat(CV2.blocks,[ntrials / k,1]);
% Creating indices for test set and training set
for i = 1: k
    test.indices(:,i) = CV.indices == i;
    train.indices(:,i) = CV.indices ~= i;
end



%% Iterative Lasso
% Try the 1st iteration
for i = 1:k
    % Split the data into training set and testing set 
    X.test = X.raw(test.indices(:,i) ,:);
    X.train = X.raw(train.indices(:,i) ,:);

    % Fit cvglmnet
    cvfit(i) = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);
    % Get the indice for the lambda with the best accuracy 
    lambda.best(i) = find(cvfit(i).lambda == cvfit(i).lambda_min);
    % Plot the cross-validation curve
%     cvglmnetPlot(cvfit(i));

    % Set the lambda value
    opts(i) = glmnetSet();
    opts(i).lambda = cvfit(i).lambda_min;

    % Fit glmnet
    fit(i) = glmnet(X.train, rowLabels.train, 'binomial', opts);

    % Evaluate the prediction 
    test.prediction(:,i) = (X.test * fit(i).beta + repmat(fit(i).a0, [test.size, 1])) > 0 ;  
    test.accuracy(:,i) = mean(rowLabels.test == test.prediction(:,i))'
    
    % Find indices for the voxels that have been used
    voxel(i).used = find (fit(i).beta ~= 0);
    % Find true signals that have been identified
    voxel(i).signal = sum(voxel(i).used <= 10);
    % Find indices for the voxels that have not been used
    voxel(i).remain = find (fit(i).beta == 0);
    % How many voxels have been used for each iteration
    voxel(i).num = sum(fit(i).beta ~= 0);

end

% Display the average accuracy for this procedure 
disp(['The mean accuracy is ' num2str(mean(test.accuracy))])
% See if it is better than chance 
ttest(test.accuracy, 0.5)
disp('number of voxels')
voxel.num
disp('number of signals')
voxel.signal


%% 1st, try the next iteration 

% New data set
for i = 1:k
    % Re-subset the dataset
    X.iter = X.raw(:,voxel(i).remain);

    % Split the data into training set and testing set 
    X.test = X.iter(test.indices(:,i) ,:);
    X.train = X.iter(train.indices(:,i) ,:);

    % Fit cvglmnet
    cvfit(i) = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);
    % Get the indice for the lambda with the best accuracy 
    lambda.best(i) = find(cvfit(i).lambda == cvfit(i).lambda_min);
    % Plot the cross-validation curve
%     cvglmnetPlot(cvfit(i));

    % Set the lambda value
    opts(i) = glmnetSet();
    opts(i).lambda = cvfit(i).lambda_min;

    % Fit glmnet
    fit(i) = glmnet(X.train, rowLabels.train, 'binomial', opts);

    % Evaluate the prediction 
    test.prediction(:,i) = (X.test * fit(i).beta + repmat(fit(i).a0, [test.size, 1])) > 0 ;  
    test.accuracy(:,i) = mean(rowLabels.test == test.prediction(:,i))'
    
    % Find indices for the voxels that have been used
    voxel(i).used = find (fit(i).beta ~= 0);
    % Find true signals that have been identified
    voxel(i).signal = sum(voxel(i).used <= 10);
    % Find indices for the voxels that have not been used
    voxel(i).remain = find (fit(i).beta == 0);
    % How many voxels have been used for each iteration
    voxel(i).num = sum(fit(i).beta ~= 0);
    
end

% Display the average accuracy for this procedure 
disp(['The mean accuracy is ' num2str(mean(test.accuracy))])
% See if it is better than chance 
ttest(test.accuracy, 0.5)