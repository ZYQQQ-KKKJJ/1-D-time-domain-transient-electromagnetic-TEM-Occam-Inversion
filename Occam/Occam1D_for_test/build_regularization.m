function R = build_regularization(nlayer)
    % 构建二阶差分正则化矩阵
    nparams = 2*nlayer - 1; % 电阻率参数 + 厚度参数
    
    % 创建二阶差分矩阵
    R = zeros(nparams-2, nparams);
    for i = 1:nparams-2
        R(i, i) = 1;
        R(i, i+1) = -2;
        R(i, i+2) = 1;
    end
end


