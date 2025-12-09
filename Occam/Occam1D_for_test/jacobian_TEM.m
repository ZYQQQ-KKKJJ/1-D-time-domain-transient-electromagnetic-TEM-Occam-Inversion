function J = jacobian_TEM(rho, thick, t, a, I)
    nlayer = length(rho);
    nparams = nlayer + length(thick);
    ntime = length(t);
    J = zeros(ntime, nparams);

    delta = 0.01;  % log空间建议小一点扰动
    TEM0 = fwd_TEM(rho, thick, t, a, I);

    % 电阻率参数扰动 (log10 空间)
    for i = 1:nlayer
        prho = rho;
        prho(i) = rho(i) * 10^delta; 
        TEM_pert = fwd_TEM(prho, thick, t, a, I);
        J(:, i) = (log10(TEM_pert) - log10(TEM0)) / delta;
    end

    % 厚度参数扰动 (log10 空间)
    for i = 1:length(thick)
        pthick = thick;
        pthick(i) = thick(i) * 10^delta;
        TEM_pert = fwd_TEM(rho, pthick, t, a, I);
        J(:, nlayer + i) = (log10(TEM_pert) - log10(TEM0)) / delta;
    end
end

