% 1D TEM Occam inversion (Forward modeling: ABFM)
% This program is currently applicable for Synthetic AND Field Model(Data). 
% If u want to use this program, maybe some modifications and adjustments for inversion parameters are required.
% This program is currently in the beta version. 
% If you find any errors or shortcomings, please feel free to contact me via email. 
% I am a Chinese student, so for some explanations in this program I used Chinese. (＾▽＾)
% Best wishes!

% Yuqi Zhao 
% yqzhao24@ mails.jlu.edu.cn
% 2025/12/09
% JiLin University, ChangChun
% ✌️ ✌️Yeah x2！
% ======================================================================= %
%% 
% This version demonstrates using tTEM20AAR as an example.
%
%% 
clc;
clear;
close all;
warning off;
%rng('default');
fontsize_value=18;
fontname_str='Times New Roman';
%% ================= 数据导入 ====================
folder_name_1='仿真实验数据\';
V_simulation=load([folder_name_1,'v_processed_tTEM20AAR.txt']);
[survey_point_size,time_size]=size(V_simulation); % 数据集份数

%% ================= 仪器参数 ====================
t =[1.044e-5,1.234e-5,1.434e-5,1.634e-5,...
    1.834e-5,2.034e-5,2.284e-5,2.584e-5,2.934e-5,3.384e-5,3.984e-5,...
    4.734e-5,5.634e-5,6.784e-5,8.234e-5,1.008e-4,1.243e-4,1.533e-4,...
    1.903e-4,2.369e-4,2.948e-4];
depth=[1.0000,1.0880,1.1840,1.2880,1.4020,1.5250,1.6600,1.8060,1.9650,2.1380,...
  2.3260,2.5310,2.7540,2.9970,3.2610,3.5480,3.8610,4.2010,4.5710,4.9740,...
  5.4120,5.8890,6.4080,6.9730,7.5870,8.2550,8.9830,9.7740,10.6390];
a=25;
I=1/500;

% 测点数量
Nsite = survey_point_size; 
layers = 30;

% 存储结果
rho_all = zeros(Nsite, layers);
thk_all = zeros(Nsite, layers-1);
misfit_all = zeros(Nsite, 1);
iter_all = zeros(Nsite, 1);
tic
for isite = 1:Nsite
    fprintf('\n=======================================\n');
    fprintf('******** 开始第 %d 个测点反演 ********\n', isite);
    fprintf('=======================================\n');

    TEM_sin = V_simulation(isite,:);
    %% ============ Occam 参数 ====================
    nlayer = layers;
    Niter = 100;
    Trms = 0.00018;
    dlambda = 1.1;

    LBR = 10 *ones(1, layers);   UBR = 1000 * ones(1, layers);
    LBT = depth;   UBT = depth;

    %% ============ 初始化 ============
    rho = mean([LBR; UBR]);
    thick = mean([LBT; UBT]);
    TEM = fwd_TEM_stable(rho, thick, t, a, I);
    E = misfit_TEM(TEM_sin, TEM);

    lambda = 1;
    best_rho = rho; best_thick = thick; best_E = E;

    E_history = zeros(1, Niter);
    lambda_history = zeros(1, Niter);

    %% ============ Occam 主循环 ============
    for iter = 1:Niter
        J = jacobian_TEM(rho, thick, t, a, I);
        d = log10(TEM_sin) - log10(TEM);
        R = build_regularization(nlayer);

        lambdar = lambda * dlambda;
        lambdal = lambda / dlambda;

        % 三个lambda搜索
        [rho_m, thk_m] = occam_update(J, d, R, lambda,   rho, thick, LBR, UBR, LBT, UBT);
        [rho_r, thk_r] = occam_update(J, d, R, lambdar, rho, thick, LBR, UBR, LBT, UBT);
        [rho_l, thk_l] = occam_update(J, d, R, lambdal, rho, thick, LBR, UBR, LBT, UBT);

        E_m = misfit_TEM(TEM_sin, fwd_TEM_stable(rho_m, thk_m, t, a, I));
        E_r = misfit_TEM(TEM_sin, fwd_TEM_stable(rho_r, thk_r, t, a, I));
        E_l = misfit_TEM(TEM_sin, fwd_TEM_stable(rho_l, thk_l, t, a, I));

        for ifind = 1:10
            if E_m <= E_l && E_m <= E_r
                break;
            elseif E_l < E_m && E_l < E_r
                rho_r = rho_m; thk_r = thk_m; lambdar = lambda;
                rho_m = rho_l; thk_m = thk_l; lambda = lambdal;
                lambdal = lambda / dlambda;
                [rho_l, thk_l] = occam_update(J, d, R, lambdal, rho, thick, LBR, UBR, LBT, UBT);
                E_m = misfit_TEM(TEM_sin, fwd_TEM_stable(rho_m, thk_m, t, a, I));
            elseif E_r < E_m && E_r < E_l
                rho_l = rho_m; thk_l = thk_m; lambdal = lambda;
                rho_m = rho_r; thk_m = thk_r; lambda = lambdar;
                lambdar = lambda * dlambda;
                [rho_r, thk_r] = occam_update(J, d, R, lambdar, rho, thick, LBR, UBR, LBT, UBT);
                E_m = misfit_TEM(TEM_sin, fwd_TEM_stable(rho_m, thk_m, t, a, I));
            else
                break;
            end
        end

        rho = rho_m; thick = thk_m;
        E = E_m;
        TEM = fwd_TEM_stable(rho, thick, t, a, I);
        E_history(iter) = E;
        lambda_history(iter) = lambda;

        if E < best_E
            best_rho = rho; best_thick = thick; best_E = E;
        end

        if E <= Trms
            break;
        end
    end
    %% =========== 存储第 isite 结果 =============
    rho_all(isite,:) = best_rho;
    thk_all(isite,:) = best_thick;
    misfit_all(isite) = best_E;
    iter_all(isite) = iter;
    fprintf('>>> 第 %d 个测点完成, Misfit=%.6f\n', isite, best_E);
    end
toc;

%% ==================== 1D Figures ====================
num_show = 6;
idx = randperm(Nsite, num_show);

figure;

for ii = 1:3
    isite = idx(ii);

    rho_i = rho_all(isite,:);
    thk_i = thk_all(isite,:);
    TEM_inv = fwd_TEM_stable(rho_i, thk_i, t, a, I);
    TEM_inv(TEM_inv<=0) = eps;

    TEM_obs = V_simulation(isite,:);
    TEM_obs(TEM_obs<=0) = eps;

    subplot(2,3,ii);
    loglog(t, TEM_obs, 'k^-', 'LineWidth',1.2, 'MarkerSize',4); hold on;
    loglog(t, TEM_inv, 'r--', 'LineWidth',1.4);
    xlabel('时间(s)'); ylabel('电压(V)');
    title(['测点 ', num2str(isite)]);
    legend('观测','反演','Location','best');
    grid on;
end

for ii = 1:3
    isite = idx(ii);

    rho_i = rho_all(isite,:);
    thk_i = thk_all(isite,:);
    depth_plot = [0 cumsum(thk_i)];

    subplot(2,3,ii+3);

    stairs(rho_i, -depth_plot, 'r--', 'LineWidth', 2);

    set(gca,'YDir','normal');
    xlabel('电阻率(Ω·m)'); ylabel('深度(m)');
    title(['模型 ', num2str(isite)]);
    legend('真实','反演','Location','best');
    grid on;
end

%% ==================== 2D Figures ====================
distance = 1:survey_point_size;

distance_fine = linspace(min(distance), max(distance), 100);
depth_fine = linspace(min(depth_plot), max(depth_plot), 100);
[Distance_fine, Depth_fine] = meshgrid(distance_fine, depth_fine);

% ----------- 反演模型插值 --------------
lg_R_inv_fine = interp2(distance, depth_plot, (log10(rho_all))', ...
                        Distance_fine, Depth_fine, 'cubic');
lg_R_inv_fine = smoothdata(lg_R_inv_fine, 2, 'movmean', 1);

figure;

contourf(Distance_fine, -Depth_fine, lg_R_inv_fine, 20, 'EdgeColor','none');
colormap(gca, turbo);
c2 = colorbar;
c2.Label.String = '\fontname{SimSun}log_{10}\rho\fontname{Times}';
c2.LineWidth = 2;
title('\fontname{SimSun}Occam反演电阻率模型','FontWeight','bold');
ylabel('\fontname{SimSun}深度 (m)');
set(gca,'LineWidth',2,'FontWeight','bold');


