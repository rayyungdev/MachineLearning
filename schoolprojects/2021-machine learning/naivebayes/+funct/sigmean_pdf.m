function [dis_pdf, mu, sig] = sigmean_pdf(x, N, rn)
    mu = mean(x);
    sig = sqrt(var(x));
    dis = linspace(rn(1), rn(2), N);
    dis_pdf = normpdf(dis, mu, sig);
end
