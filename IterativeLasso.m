function results = IterativeLasso(X,rowLabels,CV,CV2,numsignal,STOPPING_RULE)
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
            cvfit(i) = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

            % Plot the cross-validation curve
    %         cvglmnetPlot(cvfit(i));

            % Set the lambda value, using the numerical best
            opts(i) = glmnetSet();
            opts(i).lambda = cvfit(i).lambda_min;

            % Fit glmnet
            fit(i) = glmnet(X.train, rowLabels.train, 'binomial', opts);

            % Evaluate the prediction 
            %   test.prediction = X.test * ß (weight) + a (intercept)
%             test.prediction = bsxfun(@plus, X.test * fit(i).beta, fit(i).a0) > 0 ;
            test.prediction(:,i) = (X.test * fit(i).beta + repmat(fit(i).a0, [test.size, 1])) > 0 ;  
            test.accuracy(:,i) = mean(rowLabels.test == test.prediction(:,i))';

            % Recording voxels that have been used (chris' method)
            used( i, ~used(i,:) ) = fit(i).beta ~= 0;

        end


        % Take a snapshot, find out which voxels were being used currently
        USED{numIter} = used;

        % Record the results, including 
        % 1) hit.num: how many voxels that carrying true signal have been selected
        % 2) hit.rate: the porprotion of voxels have been selected 
        % 3) hit.accuracy: the accuracy for the correspoinding cv block
        % 4) hit.all : how many voxel that have been selected
        hit.num(numIter,:) = sum(used(:,1:numsignal),2); 
        hit.rate(numIter,:) = sum(used(:,1:numsignal),2) / numsignal; 
        hit.accuracy(numIter, :) = test.accuracy;
        hit.all(numIter, :) = sum(used,2);


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


    % Plot the hit rate 
    figure(3)
    plot(hit.rate,'LineWidth',1.5)
    xlabel({'Iterations'; ' ' ;...
        '* Each line indicates a different CV blocks' ;...
        '* the last two iterations were insignificant '},...
        'FontSize',12);
    ylabel('Proportion (%)', 'FontSize',12);
    title ({'The Proportion of signal carrying voxels that were selected' }, 'FontSize',12);
    axis([1 size(hit.rate(:,1),1) 0 1])
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

        % Fit cvglmnet, in order to find the best lambda
        cvfitFinal = cvglmnet (X.train, rowLabels.train, 'binomial', 'class', CV2.indices', 4);

        % Set the lambda value, using the numerical best
        opts = glmnetSet();
        opts.lambda = cvfitFinal.lambda_min;
        opts.alpha = 0;

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
    
    % Packaging results
    results.n_sig_iter = numSig;
    
    
end


