%% Iterative Lasso 
% This goal for this version is to complete iterative Lasso
% So it has to be able to merge all solutions 
% % % % % % % % % %
% This program needs two functions:
% 1. simData()
% 2. CVindices()


%% WARNING: This will clear the work space & variables.
clear;clc;
rng(1)
w = warning ('off','all'); % somehow it returns a lot of warning

%% You can set the parameters using the function: simData
% Please set parameters in this order
% % ntrials: number of trials (int>0, has to be divisible by k)
% % nvoxels: number of voxels (int>0)
% % k: number of folds for k-folds cross validation(int>0)
% % signal: the strength of signal (float)
% % noise: the strength of noise (float)
% % numRowLabels: number of row labels (int>1)
% % numsignal: number of voxels that carrying signals(int>0)

% simData returns the following:
% All parameters list above
% test.size: the size of the testing set
% X: a matrix(nvoxels, ntrials), contrains signals & noises
% rowLabels: contrains 3 types of rowlabels

[ntrials, nvoxels, k, signal, noise, rowLabels, numsignal, testSize, X ] =...
    simData( 150, 150, 5, 1, 1.5, 2, 20 );


%% Generating indices for Outer CV
% This function takes 2 parameters:
% % k: number of folds for k-folds cross validation(int>0)
% % ntrials: number of trials (int>0, has to be divisible by k)
% Note: these two variable should be defined, if simData has been called 

% CVindices returns the following:
% % CV: cross validation indices (column)
% % CV2: cross validation indices for the cvglmnet (column)
% % test.indices: indices for subseting testing set from X.raw
% % train. indices: indices for subseting training set from X.raw
[ CV, CV2, test, train ] = CVindices( k,ntrials )


%% Prepare for Iterative Lasso

% Create a copy for the raw data (X.raw)
% btw, I am not sure I need it...
X.copy = X.raw;
% Initialize the remaining voxels. (At 1st time, all voxels are remain)
for i = 1:k
    voxel(i).remain = (1:nvoxels)';
end
% Keeping track of the number of iteration
numIter = 1;


%% Interative Lasso
while true
    
% Start 5 folds cross validation
    for i = 1:k

        % Re-subset data every time
        X.iter = X.raw(:,voxel(i).remain);

        % Split the data into training & testing set 
        X.test = X.iter(test.indices(:,i) ,:);
        X.train = X.iter(train.indices(:,i) ,:);

        % Fit cvglmnet
        cvfit(i) = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

        % Get the indice for the lambda with the best accuracy 
        lambda.best(i) = find(cvfit(i).lambda == cvfit(i).lambda_min);
        % Plot the cross-validation curve
%         cvglmnetPlot(cvfit(i));

        % Set the lambda value, using the numerical best
        opts(i) = glmnetSet();
        opts(i).lambda = cvfit(i).lambda_min;

        % Fit glmnet
        fit(i) = glmnet(X.train, rowLabels.train, 'binomial', opts);

        % Evaluate the prediction 
        %   test.prediction = X.test * ß (weight) + a (intercept)
        test.prediction(:,i) = (X.test * fit(i).beta + repmat(fit(i).a0, [testSize, 1])) > 0 ;  
        test.accuracy(:,i) = mean(rowLabels.test == test.prediction(:,i))';

        % Find indices for the voxels that have been used
        voxel(i).used = find (fit(i).beta ~= 0);
        % Find indices for the voxels that have not been used
        voxel(i).remain = find (fit(i).beta == 0);    

%         % How many voxels have been used for each iteration?
%         voxel(i).num = sum(fit(i).beta ~= 0);    
%         % How many of them are true signals? (only works for 1st iter, as numsignal doesn't update)
%         voxel(i).signal = sum(voxel(i).used <= numsignal);

    end
    
    
    
    % Store all the solutions 
    i = 1;
    for j = 1 + k * (numIter - 1) : k * numIter
        voxel(j).merge = voxel(i).used;
        i = i + 1;
    end
        
    
    
    
    %% Printing some results
    disp('==============================')
    
    % Keep track of the number of iteration.
    disp(['Iteration number: ' num2str(numIter)]);
    numIter = numIter + 1;
    
    % Display the average accuracy for this procedure 
    disp(['The accuracy for each CV: ' num2str(test.accuracy) ] );
    disp(['The mean accuracy: ' num2str(mean(test.accuracy))]);
    % Test classification accuracy aganist chance 
    [t,p] = ttest(test.accuracy, 0.5);
    disp(['Result for the t-test: ' num2str(t) ',  P value: ' num2str(p)])
    % disp('number of voxels')
    % voxel.num
    % disp('number of signals')
    % voxel.signal    

    % Stop iteration, when the decoding accuracy is not better than chance
    if ttest(test.accuracy, 0.5) == 0
        disp('==============================')
        disp('Iterative Lasso was terminated, as the classification accuracy is at chance level.')
        break

    end 
    
    
end

