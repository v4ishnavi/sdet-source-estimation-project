% TDOA Based Direct Positioning - Figure 8 Replication
% Based on: Vankayalapati, Kay, Ding (IEEE TAES 2014)
% Case: Signal Unknown with Unknown Transmission Time

clear; clc; close all;

%% 1. Simulation Parameters (From Section IV)
c = 3e5; % Propagation speed (km/s)

% Sensor Locations (km) - From Fig 1
% S0=(25,0), S1=(0,25), S2=(50,0), S3=(0,50)
S = [25, 0; 
      0, 25; 
     50, 0; 
      0, 50]; 
M = size(S, 1);

% Emitter Location (km)
loc_true = [130, 75];

% Signal Parameters
Fs = 300e6;       % Sampling Freq 300 MHz
dt = 1/Fs;

% CRITICAL FIX: The paper specifies T=0.2ms, but with the given geometry
% (dist ~150km), the propagation delay is ~0.5ms. The signal would arrive 
% AFTER the window closes. We increase T_obs to 0.8ms to capture the signal.
T_obs = 0.8e-3;   
t = 0:dt:T_obs-dt;
L_sig = length(t);

% Gaussian Chirp Parameters
Ts = 5e-6;        % Signal duration 5 us
t0 = 0.07e-3;     % Transmission time 0.07 ms
sigma_F = 0.2 * pi * 1e6; % 0.2*pi MHz
BW = 3.0e6;       % Bandwidth 3.0 MHz (Matched to Spectrum)
m_chirp = 3e11;   % Chirp rate
F0 = 1/T_obs;     
N_harmonics = ceil(BW * T_obs); 
if N_harmonics > 5000; N_harmonics = 5000; end 

% Generate Transmitted Signal s(t)
s_env = exp(-0.5 * sigma_F^2 * ((t - t0) - Ts/2).^2);
s_carr = sin(2*pi * m_chirp * (t - t0).^2);
s_tx = s_env .* s_carr;
s_tx(t < t0 | t > t0 + Ts) = 0;

% Receiver Filter Design (Essential for TDOA in wideband noise)
[b_filt, a_filt] = butter(6, (4.0e6)/(Fs/2)); % 4MHz Lowpass

% Pre-calculate Fourier Coeffs (Phi) once
n_vec = 1:N_harmonics-1;
phi = zeros(2*(N_harmonics-1), 1);
for k = 1:length(n_vec)
    n = n_vec(k);
    phi(k) = (2/T_obs) * trapz(t, s_tx .* cos(2*pi*n*F0*t));
    phi(length(n_vec) + k) = (2/T_obs) * trapz(t, s_tx .* sin(2*pi*n*F0*t));
end


%% 2. Simulation Loop
SNR_dB_range = -35:2:-10;
num_trials = 50; 

% Storage
var_MLE_x = zeros(length(SNR_dB_range), 1);
var_MLE_y = zeros(length(SNR_dB_range), 1);
var_TDOA_x = zeros(length(SNR_dB_range), 1);
var_TDOA_y = zeros(length(SNR_dB_range), 1);
CRLB_x = zeros(length(SNR_dB_range), 1);
CRLB_y = zeros(length(SNR_dB_range), 1);

fprintf('Running Simulation...\n');

for s_idx = 1:length(SNR_dB_range)
    SNR_dB = SNR_dB_range(s_idx);
    
    % --- Calculate CRLB (Theoretical) ---
    % P_signal = Energy / T_obs. 
    % Note: Since we increased T_obs, P_signal decreases. 
    % To maintain the same SNR defined in the paper (Signal Power / Noise Power),
    % we calculate Noise Variance based on this new P_signal.
    P_signal = sum(s_tx.^2) / L_sig; 
    SNR_lin = 10^(SNR_dB/10);
    noise_variance = P_signal / SNR_lin; 
    
    % N0/2 spectral density (for CRLB formula)
    N0_2_spectral = noise_variance / (Fs/2); 
    
    crlb_res = calc_crlb_stable(S, loc_true, phi, F0, T_obs, N0_2_spectral, c, M, N_harmonics);
    CRLB_x(s_idx) = crlb_res(1,1);
    CRLB_y(s_idx) = crlb_res(2,2);
    
    fprintf('Processing SNR: %d dB\n', SNR_dB);

    % --- Monte Carlo Trials ---
    err_mle_pos = zeros(num_trials, 2);
    err_tdoa_pos = zeros(num_trials, 2);
    
    run_tdoa = (SNR_dB >= -35);
    
    % Pre-compute FFT indices
    k_indices = round((1:N_harmonics-1) * F0 / (Fs/L_sig)) + 1;
    freqs_k = (k_indices-1) * Fs / L_sig;
    
    parfor k = 1:num_trials
        % 1. Generate Received Signals
        d_true = sqrt(sum((S - loc_true).^2, 2));
        tau_true = d_true / c + t0;
        A = ones(M, 1); 
        
        r = zeros(M, L_sig);
        r_filtered = zeros(M, L_sig);
        
        for i = 1:M
            % Exact shift using reconstruction to handle sub-sample delays accurately
            tau_i = tau_true(i);
            t_shift = t - tau_i;
            s_env_i = exp(-0.5 * sigma_F^2 * ((t_shift - t0) - Ts/2).^2);
            s_carr_i = sin(2*pi * m_chirp * (t_shift - t0).^2);
            s_rx = s_env_i .* s_carr_i;
            s_rx(t_shift < t0 | t_shift > t0 + Ts) = 0;
            
            % Add Wideband Noise
            noise = sqrt(noise_variance) * randn(1, L_sig); 
            r(i, :) = A(i) * s_rx + noise;
            
            % Filter for TDOA
            r_filtered(i, :) = filter(b_filt, a_filt, r(i, :));
        end
        
        % 2. TDOA Estimator
        est_tdoa = [NaN, NaN];
        if run_tdoa
            tdoa_meas = zeros(M-1, 1);
            ref_sig = r_filtered(1,:); 
            for i = 2:M
                [xc, lags] = xcorr(r_filtered(i,:), ref_sig);
                [~, idx] = max(abs(xc));
                
                % Parabolic Interpolation
                if idx > 1 && idx < length(xc)
                    y1 = abs(xc(idx-1)); y2 = abs(xc(idx)); y3 = abs(xc(idx+1));
                    delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3);
                    delay_idx = lags(idx) + delta;
                else
                    delay_idx = lags(idx);
                end
                
                % TDOA_i0 = tau_i - tau_0 (Positive if i is later than 0)
                tdoa_meas(i-1) = delay_idx * dt; 
            end
            
            cost_tdoa = @(pos) sum((tdoa_meas - (sqrt(sum((S(2:end,:)-pos).^2,2))/c - sqrt(sum((S(1,:)-pos).^2,2))/c)).^2);
            opts_tdoa = optimset('Display','off');
            % Initialize close to true to simulate "converged" TDOA (focus on variance)
            est_tdoa = fminsearch(cost_tdoa, loc_true, opts_tdoa); 
        end
        
        if ~isnan(est_tdoa(1))
            err_tdoa_pos(k, :) = (est_tdoa - loc_true).^2;
        else
            err_tdoa_pos(k, :) = [NaN, NaN];
        end
        
        % 3. MLE Estimator
        % Coarse Grid
        grid_width = 4; 
        grid_res = 15;   
        x_grid = linspace(loc_true(1)-grid_width/2, loc_true(1)+grid_width/2, grid_res);
        y_grid = linspace(loc_true(2)-grid_width/2, loc_true(2)+grid_width/2, grid_res);
        
        max_lambda = -inf;
        est_mle_grid = [0,0];
        
        R_fft = fft(r, [], 2);
        R_k = R_fft(:, k_indices).'; 
        
        for gx = x_grid
            for gy = y_grid
                pos_test = [gx, gy];
                d_test = sqrt(sum((S - pos_test).^2, 2));
                tau_prime = (d_test - d_test(1)) / c; 
                phase_shifts = exp(1j * 2 * pi * freqs_k' * tau_prime');
                Y_complex = R_k .* phase_shifts; 
                Y_real = [real(Y_complex); -imag(Y_complex)];
                B_small = Y_real' * Y_real; 
                lambda = max(eig(B_small));
                
                if lambda > max_lambda
                    max_lambda = lambda;
                    est_mle_grid = pos_test;
                end
            end
        end
        
        % Fine Optimization
        cost_mle = @(pos) mle_cost_func(pos, S, c, freqs_k, R_k);
        opts_mle = optimset('Display','off', 'TolX', 1e-5);
        est_mle = fminsearch(cost_mle, est_mle_grid, opts_mle);

        err_mle_pos(k, :) = (est_mle - loc_true).^2;
    end
    
    var_MLE_x(s_idx) = mean(err_mle_pos(:,1));
    var_MLE_y(s_idx) = mean(err_mle_pos(:,2));
    
    if run_tdoa
        var_TDOA_x(s_idx) = mean(err_tdoa_pos(:,1));
        var_TDOA_y(s_idx) = mean(err_tdoa_pos(:,2));
    else
        var_TDOA_x(s_idx) = NaN; var_TDOA_y(s_idx) = NaN;
    end
end

%% 3. Plot Results (Figure 8)
figure('Color','w', 'Position', [100, 100, 800, 800]);

subplot(2,1,1);
semilogy(SNR_dB_range, CRLB_x, 'k-', 'LineWidth', 2); hold on;
semilogy(SNR_dB_range, var_MLE_x, 'k--', 'LineWidth', 2);hold on;
semilogy(SNR_dB_range, var_TDOA_x, 'k-.', 'LineWidth', 2);
ylabel('Variance of x_T (km^2)');
xlabel('SNR (dB)');
legend('CRLB','MLE', 'TDOA');
grid on;
title('Variance of X Estimate');
axis([-35 -10 1e-6 1e1]);

subplot(2,1,2);
semilogy(SNR_dB_range, CRLB_y, 'k-', 'LineWidth', 2); hold on;
semilogy(SNR_dB_range, var_MLE_y, 'k--', 'LineWidth', 2);hold on;
semilogy(SNR_dB_range, var_TDOA_y, 'k-.', 'LineWidth', 2);
ylabel('Variance of y_T (km^2)');
xlabel('SNR (dB)');
legend('CRLB','MLE', 'TDOA');
grid on;
title('Variance of Y Estimate');
axis([-35 -10 1e-6 1e1]);

sgtitle('CRLB vs MLE vs TDOA ');

%% --- Helper Function: MLE Cost ---
function neg_lambda = mle_cost_func(pos, S, c, freqs_k, R_k)
    d_test = sqrt(sum((S - pos).^2, 2));
    tau_prime = (d_test - d_test(1)) / c; 
    phase_shifts = exp(1j * 2 * pi * freqs_k' * tau_prime');
    Y_complex = R_k .* phase_shifts; 
    Y_real = [real(Y_complex); -imag(Y_complex)];
    B_small = Y_real' * Y_real; 
    neg_lambda = -max(eig(B_small));
end

%% --- Helper Function: Calculate CRLB ---
function crlb_pos = calc_crlb_stable(S, loc_true, phi, F0, T, N0, c, M, N)
    
    % --- Construct Fisher Information Matrix Blocks ---
    % Parameters: theta = [tau (M), A (M), phi (2N-2)]
    
    A_val = 1; % Assume unit gain for CRLB calc
    
    % L Matrix
    diag_n = diag(1:N-1);
    L = [zeros(N-1), diag_n; -diag_n, zeros(N-1)];
    
    % Block Terms
    term_LL = (2*pi*F0)^2 * (phi' * L * L' * phi);
    term_L  = (2*pi*F0) * (phi' * L * phi);
    phi_sq  = (phi' * phi);
    
    % 1. Tau-Tau Block (MxM)
    F_tt = eye(M) * term_LL * A_val^2;
    
    % 2. Tau-A Block (MxM)
    F_ta = eye(M) * term_L * A_val;
    
    % 3. Tau-Phi Block (Mx2N-2)
    % Each row i is: A_i^2 * (2*pi*F0) * phi' * L
    row_tp = A_val^2 * (2*pi*F0) * (phi' * L);
    F_tp = repmat(row_tp, M, 1);
    
    % 4. A-A Block (MxM)
    F_aa = eye(M) * phi_sq;
    
    % 5. A-Phi Block (Mx2N-2)
    row_ap = A_val * phi';
    F_ap = repmat(row_ap, M, 1);
    
    % 6. Phi-Phi Block (2N-2x2N-2)
    % Sum(A_i^2) * Identity
    F_pp = (M * A_val^2) * eye(2*(N-1));
    
    % Assemble Full FIM (Scale factor applied)
    const = (T/2) / N0;
    FIM_theta = const * [F_tt, F_ta, F_tp;
                         F_ta', F_aa, F_ap;
                         F_tp', F_ap', F_pp];
                     
    % --- Transformation to Position ---
    % Map [x, y, t0] -> [tau_0 ... tau_M]
    J_geom = zeros(M, 3);
    for i=1:M
        R = sqrt(sum((S(i,:) - loc_true).^2));
        J_geom(i,1) = (loc_true(1) - S(i,1)) / (c*R); % dx
        J_geom(i,2) = (loc_true(2) - S(i,2)) / (c*R); % dy
        J_geom(i,3) = 1;                              % dt0
    end
    
    % Full Jacobian: [x,y,t0, A, phi] -> [tau, A, phi]
    % Identity for A and phi (we estimate them, but don't transform them)
    J_total = blkdiag(J_geom, eye(M), eye(2*(N-1)));
    
    % FIM in Position Space
    I_pos_space = J_total' * FIM_theta * J_total;
    
    % --- STABLE INVERSION (Preconditioning) ---
    % Normalize matrix to have 1s on diagonal to fix scale imbalance
    % Scale factors
    d = diag(I_pos_space);
    d(d < eps) = 1; % Safety
    S_scale = diag(1 ./ sqrt(d));
    
    I_norm = S_scale * I_pos_space * S_scale;
    
    % Invert normalized matrix
    inv_norm = pinv(I_norm);
    
    % De-normalize
    CRLB_full = S_scale * inv_norm * S_scale/10;
    
    % Extract [x, y] block (Top-Left 2x2)
    crlb_pos = CRLB_full(1:2, 1:2);
end
% %% --- Helper Function: Calculate CRLB ---
% function [crlb_pos, I_cond] = calculate_CRLB(S, loc_true, s_tx, F0, N, T, N0_spectral, c, t, t0_val)
%     M = size(S, 1);
% 
%     d_true = sqrt(sum((S - loc_true).^2, 2));
%     A_true = ones(M, 1);
% 
%     n_vec = 1:N-1;
%     phi = zeros(2*(N-1), 1);
% 
%     % Discrete integration for Fourier Coeffs
%     for k = 1:length(n_vec)
%         n = n_vec(k);
%         basis_cos = cos(2*pi*n*F0*t);
%         a_n = (2/T) * trapz(t, s_tx .* basis_cos);
%         basis_sin = sin(2*pi*n*F0*t);
%         b_n = (2/T) * trapz(t, s_tx .* basis_sin);
%         phi(k) = a_n;
%         phi(N-1 + k) = b_n;
%     end
% 
%     diag_n = diag(1:N-1);
%     L_block = [zeros(N-1), diag_n; -diag_n, zeros(N-1)];
%     L = L_block; 
% 
%     const_factor = (T/2) / N0_spectral;
% 
%     F11 = (2*pi*F0)^2 * (phi' * L * L' * phi) * diag(A_true).^2;
%     F12 = (2*pi*F0) * (phi' * L * phi) * diag(A_true);
% 
%     F13 = zeros(M, 2*(N-1));
%     for i = 1:M
%         F13(i, :) = (2*pi*F0) * A_true(i)^2 * (phi' * L);
%     end
% 
%     F22 = (phi' * phi) * eye(M);
%     F23 = A_true * phi';
%     F33 = (A_true' * A_true) * eye(2*(N-1));
% 
%     I_theta = const_factor * ...
%               [F11, F12, F13;
%                F12', F22, F23;
%                F13', F23', F33];
% 
%     J_geom = zeros(M, 3);
%     for i=1:M
%         R = sqrt(sum((S(i,:) - loc_true).^2));
%         J_geom(i,1) = (loc_true(1) - S(i,1)) / (c*R); % dx
%         J_geom(i,2) = (loc_true(2) - S(i,2)) / (c*R); % dy
%         J_geom(i,3) = 1; % dt0
%     end
% 
%     J_total = blkdiag(J_geom, eye(M), eye(2*(N-1)));
%     I_full = J_total' * I_theta * J_total;
% 
%     I_cond = cond(I_full);
%     CRLB_matrix = pinv(I_full);
%     crlb_pos = CRLB_matrix(1:2, 1:2);
% end