clear; 
rng(2);

%% Set parameters here
iteration = 100; 
signal = 0;

textprogressbar('calculating: ');

for j = 1:4
    signal = signal + 0.25;
    

    for i = 1:iteration
        textprogressbar(sub2ind([iteration,4],i,j));
        %% Setup
        X.all = randn(100,200);

        X.all(1:50,1:20) = X.all(1:50,1:20) + signal;

        Y.all = false(100,1);
        Y.all(1:50) = true;

        CV = mod(0:99,5) == 0;

        X.test = X.all(CV,:);
        X.train = X.all(~CV,:);
        Y.test = Y.all(CV);
        Y.train = Y.all(~CV);


        %% Lasso
        opts = glmnetSet();

        cvind = mod((1:80)-1,4) + 1;

        fitObj_cv = cvglmnet(X.train,Y.train,'binomial',opts,'class',4,cvind);

        opts.lambda = fitObj_cv.lambda_min;

        fitObj_lasso = glmnet(X.train, Y.train, 'binomial',opts);

        err.LASSO(j,i) = glmnet_err(fitObj_lasso, X.train, Y.train);


        %% subset X
        X.test = X.test(:,fitObj_lasso.beta ~= 0);
        X.train = X.train(:,fitObj_lasso.beta ~= 0);

        if size(X.train,2) == 0
            err.glm(j,i) = mean(Y.test==0);
            err.RIDGE(j,i) = mean(Y.test==0);
            continue
        end


        %% GLM 
        temp = glmfit(X.train,Y.train,'binomial');
        fitObj_glm.a0 = temp(1);
        fitObj_glm.beta = temp(2:end);

        err.glm(j,i) = glmnet_err(fitObj_glm, X.train, Y.train);


        %% Ridge
        opts = glmnetSet();
        opts.alpha = 0;

        cvind = mod((1:80)-1,4) + 1;
        fitObj_cv_r = cvglmnet(X.train,Y.train,'binomial',opts,'class',4,cvind);
        opts.lambda = fitObj_cv_r.lambda_min;

        fitObj_ridge = glmnet(X.train, Y.train, 'binomial',opts);

        err.RIDGE(j,i) = glmnet_err(fitObj_ridge, X.train, Y.train);


    end
    
end


mean(err.LASSO,2)
mean(err.glm,2)
mean(err.RIDGE,2)

plot([0.25:0.25:1],mean(err.LASSO,2), 'b','LineWidth',1.5 )
hold on 
plot([0.25:0.25:1],mean(err.glm,2), 'g','LineWidth',1.5)
plot([0.25:0.25:1],mean(err.RIDGE,2), 'r','LineWidth',1.5)

legend('LASSO', 'GLM','RIDGE',...
    'Location','NorthEast')
title('Comparing Lasso, ridge & glm','FontSize',12)
xlabel('Strength of the Signal','FontSize',12);
ylabel('Error','FontSize',12);
set(gca,'xtick',0.25:0.25:1)
hold off