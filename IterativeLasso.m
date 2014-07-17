function [results, hit , final, used] = IterativeLasso(X,rowLabels,CV,CV2,numsignal,STOPPING_RULE)
    %% Prepare for Iterative Lasso
    nvoxels = size(X.raw,2);
    k = max(CV.blocks);
    test.size = size(X.raw,1)/k;
    test.indices = bsxfun(@eq,CV.indices,1:5);
    train.indices = ~bsxfun(@eq,CV.indices,1:5);
    
    
    % Keeping track of the number of iteration
    numIter = 0;
    % Keeping track of number of significant iteration
    numSig = 0;
    % Counter for Stopping criterion 
    % ps: the loop stops when t-test insig. twice
    counter = 0;
    % Create a matrix to index voxels that have been used (Chris' method)
    used = false(k,nvoxels); 



    %% Interative Lasso

    % Infinite loop, until reach stopping crterion
    while true

        numIter = numIter + 1;

    % Start 5 folds cross validation
        for i = 1:k

            % Re-subset data every time
            X.iter = X.raw(:,~used(i,:));

            % Split the data into training & testing set 
            X.test = X.iter(test.indices(:,i) ,:);
            X.train = X.iter(train.indices(:,i) ,:);

            % Fit cvglmnet
            cvfit = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

            % Plot the cross-validation curve
    %         cvglmnetPlot(cvfit(i));

            % Set the lambda value, using the numerical best
            opts = glmnetSet();
            opts.lambda = cvfit.lambda_min;

            % Fit glmnet
            fit = glmnet(X.train, rowLabels.train, 'binomial', opts);

            % Evaluate the prediction 
            %   test.prediction = X.test * ß (weight) + a (intercept)
            test.prediction(:,i) = (X.test * fit.beta + repmat(fit.a0, [test.size, 1])) > 0 ;  
            test.accuracy(:,i) = mean(rowLabels.test == test.prediction(:,i))';

            % Releveling
            opts.alpha = 0;
            fitRidge = glmnet(X.train, rowLabels.train, 'binomial', opts);
            r.prediction(:,i) = (X.test * fitRidge.beta + repmat(fitRidge.a0, [test.size, 1])) > 0 ;  
            r.accuracy(:,i) = mean(rowLabels.test == r.prediction(:,i))';
            
            
            % Recording voxels that have been used (chris' method)
            used( i, ~used(i,:) ) = fit.beta ~= 0;

        end


        % Take a snapshot, find out which voxels were being used currently
        USED{numIter} = used;

        % Record the results, including 
        % 1) hit.numSignal: how many voxels that carrying true signal have been selected
        hit.numSignal(numIter,:) = sum(used(:,1:numsignal),2); 
        % 2) hit.rate: the porprotion of voxels have been selected         
        hit.rate(numIter,:) = sum(used(:,1:numsignal),2) / numsignal; 
        % 3) hit.accuracy: the accuracy for the correspoinding cv block        
        hit.accuracy(numIter, :) = test.accuracy;
        % 4) hit.all : how many voxel that have been selected (cumulative)        
        hit.all(numIter, :) = sum(used,2);
        % 5) hit.num_current: how many voxel that have been selected in the
        % current iteration
        if numIter == 1
            hit.num_current(1,:) = hit.all(1,:);
        else
            hit.num_current(numIter,:) = hit.all(numIter,: ) - hit.all(numIter - 1,:) ;
        end
        % 6) hit.rate_current: how many signal-carrying voxels were
        % selected in current iteration
        hit.rate_current(1,:) = hit.rate(1,:);
        for i = 2:size(hit.rate,1)
            hit.rate_current(i,:) = hit.rate(i,:) - hit.rate(i-1,:);
        end
        % 5) hit.ridgeAccuracy: the accuracy for ridge regression
        hit.ridgeAccuracy(numIter, :) = r.accuracy;       


        %% Printing some results
        disp('==============================')

        % Keep track of the number of iteration.
        disp(['Iteration number: ' num2str(numIter)]);

        % Display the average accuracy for this procedure 
    %     disp(['The accuracy for each CV: ' num2str(test.accuracy) ] );
        disp(['The mean accuracy: ' num2str(mean(test.accuracy))]);
        % Test classification accuracy aganist chance 
        [t,p] = ttest(test.accuracy, 0.5);
        
        if t == 1 % t could be NaN
            numSig = numSig + 1;
            disp(['Result for the t-test: ' num2str(t) ',  P = ' num2str(p), ' *']) 
        else
            disp(['Result for the t-test: ' num2str(t) ',  P = ' num2str(p)]) 
        end
        
        % how many voxel were selected in the current iteration: 
        disp('voxels that were selected in the current iteration:')
        disp(hit.num_current(numIter,:))
        disp('voxels selected (cumulative): ')
        disp(hit.all(numIter,:))
        
        % Stop iteration, when the decoding accuracy is not better than chance
        if t ~= 1   %  ~t will rise a bug, when t is NaN
            counter = counter + 1;

            if counter == STOPPING_RULE;
            % stop, if t-test = 0 twice
                disp(' ')
                disp('* Iterative Lasso was terminated, as the classification accuracy is at chance level twice.')
                disp(' ')
                break
            end

        else
            counter = 0;
        end 


    end

    disp('Here are the accuracies for each iteration: ')
    disp('(row: iteration; colum: CV)')
    disp(hit.accuracy)
    disp('Mean accuracies:')
    disp(mean(hit.accuracy,2)) 


    % Plot the hit.rate 
    figure(3)
    subplot(1,2,1)
    plot(hit.rate,'LineWidth',1.5)
    xlabel({'Iterations'; ' ' ;...
        '* Each line indicates a different CV blocks' ;...
        '* the last two iterations were insignificant '},...
        'FontSize',12);
    ylabel('Proportion of signal-carrying voxels (%)', 'FontSize',12);
    title ({'The Proportion of signal carrying voxels that were selected over time' }, 'FontSize',12);
    axis([1 size(hit.rate(:,1),1) 0 1])
    set(gca,'xtick',1:size(hit.rate(:,1),1))
    
    % plot the hit.rate_current
    subplot(1,2,2)
    plot(hit.rate_current)
        xlabel({'Iterations'; ' ' ;...
        '* Each line indicates a different CV blocks' ;...
        '* the last two iterations were insignificant '},...
        'FontSize',12);
    title ({'The Proportion of signal carrying voxels that were selected in current iteration' }, 'FontSize',12);
    axis([1 size(hit.rate_current(:,1),1) 0 1])
    set(gca,'xtick',1:size(hit.rate(:,1),1))

    % plot the number of voxels selected
    figure(4)
    subplot(1,2,1)
    plot(hit.all,'LineWidth',1.5)
    xlabel({'Iterations'; ' ' ;...
        '* Each line indicates a different CV blocks' ;...
        '* the last two iterations were insignificant '},...
        'FontSize',12);
    ylabel('Number of voxels', 'FontSize',12);
    title ({'The number of voxels that were selected that were selected (cumulative)' }, 'FontSize',12);
    set(gca,'xtick',1:size(hit.rate(:,1),1))
    
    subplot(1,2,2)
    plot(hit.num_current,'LineWidth',1.5)
    xlabel({'Iterations'; ' ' ;...
        '* Each line indicates a different CV blocks' ;...
        '* the last two iterations were insignificant '},...
        'FontSize',12);
    ylabel('Number of voxels', 'FontSize',12);
    title ({'The number of voxels that were selected in single iteration' }, 'FontSize',12);
    set(gca,'xtick',1:size(hit.rate(:,1),1))
    
    %% Final step: pooling the solutions
    disp('Pooling the solutions, and fitting the final model using Ridge...')
    disp(' ')
    for i = 1:k
        % Subset: find voxels that were selected 
        X.final = X.raw( :, USED{numIter - STOPPING_RULE}(i,:) );

        % Split the final data set to testing set and training set 
        X.test = X.final(test.indices(:,i) ,:);
        X.train = X.final(train.indices(:,i) ,:);

        opts = glmnetSet();        
        opts.alpha = 0;        
        
        % Fit cvglmnet, in order to find the best lambda
        cvfitFinal = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

        % Set the lambda value, using the numerical best

        opts.lambda = cvfitFinal.lambda_min;


        % Fit glmnet 
        fitFinal = glmnet(X.train, rowLabels.train, 'binomial', opts);

        % Calculating accuracies
        final.prediction(:,i) = (X.test * fitFinal.beta + repmat(fitFinal.a0, [test.size, 1])) > 0 ;  
        final.accuracy(i) = mean(rowLabels.test == final.prediction(:,i))';


    end

    disp('Final accuracies: ')
    disp('(row: CV that just performed; colum: CV block from the iterative Lasso)')
    disp(final.accuracy)
    disp('Mean accuracy: ')
    disp(mean(final.accuracy))
    
    %% Packaging results
    results.n_sig_iter = numSig;
    results.lasso_err = mean(- (hit.accuracy(1,:) - 1),2);
    results.ridge_err = mean(- (hit.ridgeAccuracy(1,:) - 1),2);
    
end


