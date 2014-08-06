function ind = collapse_grid(dim,grid)
    d = cumprod(dim);
    x = [1,d(1:end-1)];
    ind = ((grid-1) * transpose(x)) + 1;
end