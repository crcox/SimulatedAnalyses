function err = glmnet_err(fitObj, X, Y)
    Yhat = bsxfun(@plus, X * fitObj.beta, fitObj.a0);
    Prediction = Yhat > 0;
    err = 1 - mean(Y == Prediction);
end