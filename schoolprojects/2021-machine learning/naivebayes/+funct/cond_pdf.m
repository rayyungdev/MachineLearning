function [p] = cond_pdf(X,mu, sig)
    gm = gmdistribution(mu,sig);
    p = pdf(gm, X);
end