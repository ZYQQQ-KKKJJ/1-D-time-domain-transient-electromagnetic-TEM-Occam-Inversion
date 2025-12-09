function E = misfit_TEM(obs, cal)
    % 计算RMS误差(log域)
     E = sqrt(mean((log10(obs) - log10(cal)).^2));
end