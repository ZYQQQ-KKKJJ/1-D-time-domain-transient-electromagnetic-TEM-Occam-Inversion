% 1D TEM Occam inversion (Forward modeling: ABFM)
% This program is currently only applicable for Synthetic Model(Data). 
% If u want to use it in Field Data, modifications and adjustments are required.
% This program is currently in the beta version. 
% If you find any errors or shortcomings, please feel free to contact me via email. 
% I am a Chinese student, so for some explanations in this program I used Chinese. (＾▽＾)
% Best wishes!

% Yuqi Zhao 
% yqzhao24@ mails.jlu.edu.cn
% 2025/06/25
% JiLin University, ChangChun
% ✌️ Yeah！
% ======================================================================= %
%% 
tic;
format bank
clear; clc;

% 时间序列（单位：秒）
t = logspace(-6, -3, 30); 
a = 5; I = 50;

% 真实模型参数（用于生成合成数据）
R_true = [50,300,50];
thk_true = [20,20];

% 正演模拟合成数据
TEM_sin = fwd_TEM(R_true, thk_true, t, a, I);

% 数据加噪
% noise_level = 0.15; % 15% 噪声
% noise = noise_level * randn(size(TEM_sin)) .* TEM_sin;
% TEM_sin = TEM_sin + noise;


%% 
% Occam反演参数(需要根据应用场景调节)

nlayer = 3;        % 层数
Niter = 100;         % 最大迭代次数
Trms = 0.00018;         % 目标RMS误差

dlambda = 1.1;      % 正则化参数调整因子

LBR = [1 1 1];   UBR = [300 500 300]; % resistivity bounds
LBT = [10 10];     UBT = [40 40] ;     % thickness bounds
%% 
% === 反演初始化 ===
rho = mean([LBR; UBR]);
thick = mean([LBT; UBT]);
TEM = fwd_TEM(rho, thick, t, a, I);
E = misfit_TEM(TEM_sin, TEM);

lambda = 1;
best_rho = rho; 
best_thick = thick; 
best_E = E;

E_history = zeros(1, Niter);
lambda_history = zeros(1, Niter);
%% Occam反演实现

% === Occam主循环融合 ===
% Reference ：DONG Hao
for iter = 1:Niter
    fprintf('\n=== 迭代 %d/%d ===\n', iter, Niter);
    fprintf('当前lambda: %.4f, 当前RMS: %.6f\n', lambda, E);

    J = jacobian_TEM(rho, thick, t, a, I);
    d = log10(TEM_sin) - log10(TEM);
    %d = TEM_sin-TEM;
    R = build_regularization(nlayer);
    %[R, Re, Rt] = build_regularization(nlayer, nlayer-1, 1, 0.00000001); 

    lambdar = lambda * dlambda;
    lambdal = lambda / dlambda;

    % 三个lambda值下模型
    [rho_m, thk_m] = occam_update(J, d, R, lambda, rho, thick, LBR, UBR, LBT, UBT);
    [rho_r, thk_r] = occam_update(J, d, R, lambdar, rho, thick, LBR, UBR, LBT, UBT);
    [rho_l, thk_l] = occam_update(J, d, R, lambdal, rho, thick, LBR, UBR, LBT, UBT);

    % 对应RMS
    E_m = misfit_TEM(TEM_sin, fwd_TEM(rho_m, thk_m, t, a, I));
    E_r = misfit_TEM(TEM_sin, fwd_TEM(rho_r, thk_r, t, a, I));
    E_l = misfit_TEM(TEM_sin, fwd_TEM(rho_l, thk_l, t, a, I));

    for ifind = 1:10
        if E_m <= E_l && E_m <= E_r
            fprintf('局部最优模型已找到\n'); break;
        elseif E_l < E_m && E_l < E_r
            fprintf('<< 向左搜索 <<\n');
            rho_r = rho_m; 
            thk_r = thk_m; 
            lambdar = lambda;

            rho_m = rho_l; 
            thk_m = thk_l; 
            lambda = lambdal;
            lambdal = lambda / dlambda;

            [rho_l, thk_l] = occam_update(J, d, R, lambdal, rho, thick, LBR, UBR, LBT, UBT);
            E_m = misfit_TEM(TEM_sin, fwd_TEM(rho_m, thk_m, t, a, I));

        elseif E_r < E_m && E_r < E_l
            fprintf('>> 向右搜索 >>\n');
            rho_l = rho_m; 
            thk_l = thk_m; 
            lambdal = lambda;
            rho_m = rho_r; thk_m = thk_r; lambda = lambdar;
            lambdar = lambda * dlambda;
            [rho_r, thk_r] = occam_update(J, d, R, lambdar, rho, thick, LBR, UBR, LBT, UBT);
            E_m = misfit_TEM(TEM_sin, fwd_TEM(rho_m, thk_m, t, a, I));
        else
            fprintf('无法找到最优方向，保持当前\n'); break;
        end
    end

    % 更新当前模型
    rho = rho_m; thick = thk_m;
    E = E_m;
    TEM = fwd_TEM(rho, thick, t, a, I);  
    E_history(iter) = E;
    lambda_history(iter) = lambda;

    if E < best_E
        best_rho = rho;
        best_thick = thick;
        best_E = E;
    end

    if E <= Trms
        fprintf('达到目标RMS，尝试平滑化模型\n');
        while E <= Trms && lambda < 1e5
            lambda = lambda * sqrt(dlambda);
            [rho_s, thk_s] = occam_update(J, d, R, lambda, rho, thick, LBR, UBR, LBT, UBT);
            E_s = misfit_TEM(TEM_sin, fwd_TEM(rho_s, thk_s, t, a, I));
            if E_s <= Trms
                rho = rho_s; thick = thk_s; E = E_s;
            else
                break;
            end
        end
        break;
    end
end

%% 绘图部分保持不变
% 绘图
r_plot = [0, R_true];
t_plot = [0,cumsum(thk_true),max(thk_true)*100];
r_mod = [0,best_rho];
Depth_mod = [0,cumsum(best_thick),max(best_thick)*100];
TEM_mod = fwd_TEM(best_rho, best_thick, t, a, I);


figure(1)
subplot(1,6,[1 3])
loglog(t,TEM_sin,'r',t,TEM_mod,'ob','MarkerSize',6,'LineWidth',2.5);
ylim([10^-11 10^-1])
legend({'Observed','Calculated'},'FontWeight','Bold');
xlabel('Time (s)');
ylabel('dBdt (V/m^2)');
title(['Misfit: ', num2str(best_E),' | Iter: ', num2str(iter)]);
grid on

subplot(1,6,[5 6])
stairs(r_plot,t_plot,'-b','Linewidth',2);
hold on
stairs(r_mod,Depth_mod,'--r','Linewidth',2);
axis([10^1 10^3 1 100])
legend({'True model','Inversion model'},'FontWeight','Bold','Location', 'best');
xlabel('Resistivity (\Omega.m)');
ylabel('Depth (m)');
title('Model');
set(gca,'YDir','Reverse'); set(gca, 'XScale', 'log');
grid on

figure(2)
plot(1:iter,E_history(1:iter),'r','Linewidth',1.5)
xlabel('Iteration');
ylabel('Misfit');
title('Misfit vs Iteration');
grid on

figure(3)
semilogy(1:iter,lambda_history(1:iter),'b','Linewidth',1.5)
xlabel('Iteration');
ylabel('Regularization Parameter');
title('Lambda vs Iteration');
grid on

figure(4);
imagesc(log10(abs(J)+1e-20)); colorbar;
title('Jacobian (log_{10} scale)');