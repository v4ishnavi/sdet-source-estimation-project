
clear; clc; close all;
c = 3e5;  
Fs = 300e6;
dt = 1/Fs;
T_obs = 0.2e-3;
t = (0:dt:T_obs-dt).';
N = length(t); 

Sensors = [
    0, 50;
    50, 0;
    25,0;
    0, 25;
    0,0;
];
M = size(Sensors,1);

Emitter_True = [130, 75];
sigma_F = 0.2*pi*1e6;
Ts = 5e-6;
t0 = 0.07e-3;
BW = 1.5e6;
slope = BW / Ts;

s_t = zeros(N,1);
active = t >= t0 & t <= t0+Ts;
tp = t(active) - t0;
env = exp(-0.5 * sigma_F^2 * (tp - Ts/2).^2);
chirp = sin(2*pi * slope .* tp.^2);
s_t(active) = env .* chirp;

S_fft = fft(s_t);
freqs = ((0:N-1)' - floor(N/2)) * (Fs/N);
freqs = fftshift(freqs);

ASNR_dB = -20;
Trials = 50;
Est = zeros(Trials,2);

d_true = sqrt(sum((Sensors - Emitter_True).^2,2));
tau_true = d_true / c;

disp("Running Monte Carlo...");

for k = 1:Trials
    R = zeros(N,M);
    
    for m = 1:M
        phase_shift = exp(-1j * 2*pi * freqs * tau_true(m));    % Nx1
        sig_m = ifft(S_fft .* phase_shift);                      % Nx1
        
        sig_pwr = mean(abs(sig_m).^2);
        noise_pwr = sig_pwr / 10^(ASNR_dB/10);
        noise = sqrt(noise_pwr) * randn(N,1);
        
        R(:,m) = real(sig_m) + noise;
    end
    init_guess = Emitter_True + randn(1,2)*0.2;
    cost = @(p) -dpd_cost_2d(p, R, Sensors, c, freqs);
    est_xy = fminsearch(cost, init_guess, optimset('Display','off'));

    Est(k,:) = est_xy;
end

Mean_Est = mean(Est,1);
Cov_Est = cov(Est);

disp("Finished Monte Carlo.");

figure('Color','w','Position',[200 200 800 600]);
hold on; grid on; axis equal;

scatter(Est(:,1), Est(:,2), 25, 'b', 'filled');
draw_conf_ellipse(Mean_Est, Cov_Est, 0.95);

plot(Emitter_True(1), Emitter_True(2), 'gp', 'MarkerSize',18,'MarkerFaceColor','g');
plot(Mean_Est(1), Mean_Est(2), 'rs', 'MarkerSize',18,'LineWidth',2);

scatter(Sensors(:,1), Sensors(:,2), 100, 'k', 'filled');

xlabel('X (km)', 'FontSize', 18);
ylabel('Y (km)', 'FontSize', 18);

title(sprintf('2D DPD – Scatter & 95%% Confidence Ellipse\nASNR = %d dB, Trials = %d', ...
    ASNR_dB, Trials), 'FontSize', 22, 'FontWeight','bold');

legend('Estimates','Ellipse','True','Mean','Sensors');


function J = dpd_cost_2d(pos, R, Sensors, c, freqs)

    if any(isnan(pos))
        J = -inf;
        return;
    end

    M = size(Sensors,1);
    N = size(R,1);

    d = sqrt(sum((Sensors - pos).^2,2));
    tau = d / c;

    R_fft = fft(R, [], 1);      % NxM
    Y = zeros(N,M);

    for m = 1:M
        phase = exp(1j * 2*pi*freqs * tau(m));   % Nx1
        Y(:,m) = ifft(R_fft(:,m) .* phase);
    end

    Q = real(Y' * Y);           % MxM
    J = max(eig(Q));
end



function draw_conf_ellipse(mu, Sigma, conf)
    s = chi2inv(conf, 2);
    [V,D] = eig(Sigma * s);
    t = linspace(0, 2*pi, 200);
    circ = [cos(t); sin(t)];
    ell = V * sqrt(D) * circ + mu';
    plot(ell(1,:), ell(2,:), 'r-', 'LineWidth',2);
end

% % PLOT WITHOUT SENSORS ALSO
% figure('Color','w','Position',[200 200 800 600]);
% hold on; grid on; axis equal;
% % Scatter points
% scatter(Est(:,1), Est(:,2), 25, 'b', 'filled');
% % Confidence ellipse
% draw_conf_ellipse(Mean_Est, Cov_Est, 0.95);
% % True and mean markers
% plot(Emitter_True(1), Emitter_True(2), 'gp', 'MarkerSize',18,'MarkerFaceColor','g');
% plot(Mean_Est(1), Mean_Est(2), 'rs', 'MarkerSize',18,'LineWidth',2);
% % Sensor markers
% % scatter(Sensors(:,1), Sensors(:,2), 100, 'k', 'filled');
% xlabel('X (km)');
% ylabel('Y (km)');
% title(sprintf('2D DPD – Scatter & 95%% Confidence Ellipse\nASNR = %d dB, Trials = %d', ASNR_dB, Trials));
% legend('Estimates','Ellipse','True','Mean');
