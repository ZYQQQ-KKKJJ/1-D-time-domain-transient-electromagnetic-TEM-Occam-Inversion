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

% function [R, Re, Rt] = build_regularization(nrho, nthk, w_rho, w_thk)
%     % 电阻率正则化
%     Re = zeros(nrho-2, nrho);
%     for i = 1:nrho-2
%         Re(i, i) = 1;
%         Re(i, i+1) = -2;
%         Re(i, i+2) = 1;
%     end
%     Re = w_rho * Re;
% 
%     % 厚度正则化
%     Rt = zeros(nthk-2, nthk);
%     for i = 1:nthk-2
%         Rt(i, i) = 1;
%         Rt(i, i+1) = -2;
%         Rt(i, i+2) = 1;
%     end
%     Rt = w_thk * Rt;
% 
%     R = blkdiag(Re, Rt);  % 合并成总正则
% end
