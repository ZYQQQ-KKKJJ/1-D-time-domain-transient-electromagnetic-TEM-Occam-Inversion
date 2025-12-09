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
% This version demonstrates using a 7-layer synthetic model as an example.
%
% In Part: %% ================= 数据导入 ====================
% The user may replace the default dataset using (V_simulation_original.txt).
% The file (R_simulation_original.txt) corresponds to the true resistivity values.
%
% In Part: %% ================= 仪器参数 ====================
% t refers to the time gates used in transient electromagnetic acquisition.
% Reference dataset: tTEM20AAR. // doi:10.5194/essd-13-2743-2021
% a / I represent the transmitting loop area and current amplitude,
% respectively, defining the primary excitation strength.


% In Part: %% ============ Occam 参数 ====================

% nlayer:
%   Number of layers in the 1D resistivity model.
%   In this work a 7-layer synthetic Earth structure is assumed.

% Niter:
%   Maximum number of Occam inversion iterations.
%   The algorithm will stop earlier if the misfit threshold (Trms) is reached.
%   Increasing this value allows deeper searching in the model space but increases runtime.

% Trms:
%   Target RMS misfit between observed TEM response and forward simulated data.
%   Once the RMS drops below this value, the inversion terminates.
%   A stricter value (smaller) enforces closer data fitting, but may lead to overfitting
%   or instability in the model.

% dlambda:
%   Lambda scaling factor used in the Occam line search.
%   The algorithm evaluates forward models with λ, λ/dlambda, λ*dlambda and selects the
%   lambda that gives the minimum model–data misfit with smoothness constraint.
%   Typical values range between 1.1–2.0. (maybe and need test)
%   Smaller values give slower but more stable convergence.

% LBR / UBR:
%   Lower and Upper resistivity bounds allowed during inversion.
%   These define the search space of the underground resistivity parameter.
%   In this configuration: 10 to 1000 ohm·m.
%   Setting too tight bounds may constrain the solution and prevent accurate fitting,
%   while too wide bounds can slow down convergence.

% LBT / UBT:
%   Lower and Upper bounds of layer thicknesses.
%   Here both are set equal to the initial depth array, meaning layer thicknesses are fixed.
%   If thickness variation is allowed, LBT/UBT can be expanded by ±20–50% and
%   controlled by smoothness regularization.
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
V_simulation=load([folder_name_1,'V_simulation_original.txt']);
[survey_point_size,time_size]=size(V_simulation); % 数据集份数
R_simulation_original=load([folder_name_1,'R_simulation_original.txt']);
%% ================= 仪器参数 ====================
t =[4.44e-6,6.44e-6,8.44e-6,1.044e-5,1.234e-5,1.434e-5,1.634e-5,...
    1.834e-5,2.034e-5,2.284e-5,2.584e-5,2.934e-5,3.384e-5,3.984e-5,...
    4.734e-5,5.634e-5,6.784e-5,8.234e-5,1.008e-4,1.243e-4,1.533e-4,...
    1.903e-4,2.369e-4,2.948e-4,3.683e-4,4.609e-4,5.774e-4,7.238e-4,...
    9.084e-4];
depth=[5.9620,9.0940,13.8690,21.1550,32.2690,37.6520];
a=25;
I=1;

% 测点数量
Nsite = survey_point_size; 
layers = 7;

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
    nlayer = 7;
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

    rho_true = R_simulation_original(isite,:);
    stairs(rho_true, -depth_plot, 'k', 'LineWidth', 2); hold on;
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

% ----------- 真实模型插值 --------------
lg_R_true_fine = interp2(distance, depth_plot, (log10(R_simulation_original))', ...
                         Distance_fine, Depth_fine, 'cubic');
lg_R_true_fine = smoothdata(lg_R_true_fine, 2, 'movmean', 7);

% ----------- 反演模型插值 --------------
lg_R_inv_fine = interp2(distance, depth_plot, (log10(rho_all))', ...
                        Distance_fine, Depth_fine, 'cubic');
lg_R_inv_fine = smoothdata(lg_R_inv_fine, 2, 'movmean', 7);

% ----------- 误差 ------------------------
lg_R_err = lg_R_true_fine - lg_R_inv_fine;

figure;

subplot(3,1,1);
contourf(Distance_fine, -Depth_fine, lg_R_true_fine, 20, 'EdgeColor','none');
colormap(gca, turbo);
c1 = colorbar;
c1.Label.String = '\fontname{SimSun}log_{10}\rho\fontname{Times}';
c1.LineWidth = 2;
title('\fontname{SimSun}真实电阻率模型','FontWeight','bold');
ylabel('\fontname{SimSun}深度 (m)');
set(gca,'LineWidth',2,'FontWeight','bold');

subplot(3,1,2);
contourf(Distance_fine, -Depth_fine, lg_R_inv_fine, 20, 'EdgeColor','none');
colormap(gca, turbo);
c2 = colorbar;
c2.Label.String = '\fontname{SimSun}log_{10}\rho\fontname{Times}';
c2.LineWidth = 2;
title('\fontname{SimSun}Occam反演电阻率模型','FontWeight','bold');
ylabel('\fontname{SimSun}深度 (m)');
set(gca,'LineWidth',2,'FontWeight','bold');

subplot(3,1,3);
contourf(Distance_fine, -Depth_fine, lg_R_err, 20, 'EdgeColor','none');

cmap = flipud(gray);   
colormap(gca, cmap);

c3 = colorbar;
c3.Label.String = '\fontname{SimSun}\Delta log_{10}';
c3.LineWidth = 2;

title('\fontname{SimSun}误差图','FontWeight','bold');
xlabel('\fontname{SimSun}测点序号');
ylabel('\fontname{SimSun}深度 (m)');
set(gca,'LineWidth',2,'FontWeight','bold');
