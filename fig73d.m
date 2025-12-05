

    % 1. Configuration
    c = 3e5; % Speed of light (km/s)
    Fs = 300e6; % Sampling Freq
    dt = 1/Fs;
    T_obs = 0.2e-3;
    t = 0:dt:T_obs-dt;
    N_samples = length(t); % = Fs * T_obs
    
    % 3D Sensor Geometry (5 Sensors for good 3D resolution)
    Sensors = [
         0,  0,  0;
         0, 50, 10;
        50,  0, 20;
        50, 50,  0;
        25, 25, 30
    ];
    M = size(Sensors, 1);
    
    Emitter_True = [35, 15, 12];
    
    % Signal Params (Gaussian Chirp)
    sigma_F = 0.2 * pi * 1e6;
    Ts = 5e-6;
    t0 = 0.07e-3;
    BW = 1.5e6;
    slope = BW / Ts;
    
    % Generate Base Signal
    s_t = zeros(size(t));
    active_idx = t >= t0 & t <= (t0 + Ts);
    t_pulse = t(active_idx) - t0;
    env = exp(-0.5 * sigma_F^2 * (t_pulse - Ts/2).^2);
    chirp = sin(2*pi * slope * t_pulse.^2);
    s_t(active_idx) = env .* chirp;
    
    % Pre-compute Frequency vector
    freqs = [0:ceil(N_samples/2)-1, -floor(N_samples/2):-1]' * (Fs/N_samples);
    R_spec_base = fft(s_t);

    % 2. Monte Carlo Simulation
    ASNR_dB = -20; % Low SNR as per Figure 7 in paper
    N_trials = 50; % Number of trials (Increase to 300 for smoother ellipsoid)
    
    Estimates = zeros(N_trials, 3);
    
    fprintf('Running %d Monte Carlo trials at %d dB...\n', N_trials, ASNR_dB);
    
    % True Delays
    dist_true = sqrt(sum((Sensors - Emitter_True).^2, 2));
    tau_true = dist_true / c;

    wb = waitbar(0, 'Running Simulations...');
    
    for n = 1:N_trials
        waitbar(n/N_trials, wb, sprintf('Trial %d/%d', n, N_trials));
        
        % A. Generate Noisy Signal for this trial
        r_trial = zeros(N_samples, M);
        for i = 1:M
            % Signal Phase Shift
            phase_shift = exp(-1j * 2 * pi * freqs * tau_true(i));
            sig_delayed = ifft(R_spec_base .* phase_shift');
            
            % Noise
            sig_pwr = mean(abs(sig_delayed).^2);
            noise_pwr = sig_pwr / 10^(ASNR_dB/10);
            noise = sqrt(noise_pwr) * randn(size(sig_delayed));
            
            r_trial(:, i) = real(sig_delayed) + noise;
        end
        
        % B. Estimation (DPD)
        % Using fminsearch for speed in MC. 
        % In practice, use grid search first to find basin of attraction.
        % We initialize slightly off-center to prove convergence.
        init_guess = Emitter_True + randn(1,3)*0.5; 
        
        % Optimization options
        opts = optimset('Display','off', 'TolX', 1e-4);
        
        % Cost function: Negative of Max Eigenvalue (we want to Maximize it)
        cost_func = @(pos) -dpd_cost(pos, r_trial, Sensors, c, freqs);
        
        [est_pos, ~] = fminsearch(cost_func, init_guess, opts);
        % disp(est_pos);
        Estimates(n, :) = est_pos;
    end
    close(wb);

    %% 3. Analysis and Ellipsoid Calculation
    Mean_Est = mean(Estimates);
    Cov_Est = cov(Estimates);
    
    fprintf('\nResults:\n');
    fprintf('True Pos: [%0.2f, %0.2f, %0.2f]\n', Emitter_True);
    fprintf('Mean Est: [%0.2f, %0.2f, %0.2f]\n', Mean_Est);
    fprintf('Covariance Diagonal (Var X, Y, Z): %e, %e, %e\n', diag(Cov_Est));

    %% 4. Plotting (3D Figure 7)
    figure('Color','w', 'Position', [100, 100, 800, 600]);
    hold on; grid on; axis equal;
    
    % A. Scatter Plot
    scatter3(Estimates(:,1), Estimates(:,2), Estimates(:,3), 15, 'b', 'filled');
    
    % B. Plot 95% Confidence Ellipsoid
    plot_confidence_ellipsoid(Mean_Est, Cov_Est, 0.95);
    
    % C. Plot True Position
    h_true = scatter3(Emitter_True(1), Emitter_True(2), Emitter_True(3), 100, 'g', 'filled', 'p');
    h_mean = scatter3(Mean_Est(1), Mean_Est(2), Mean_Est(3), 100, 'k', 'filled', 's');
    
    xlabel('X (km)'); ylabel('Y (km)'); zlabel('Z (km)');
    title(sprintf('3D Extension of Fig. 7: MLE Scatter & 95%% Conf. Ellipsoid\n(ASNR = %d dB, N = %d)', ASNR_dB, N_trials));
    legend([h_true, h_mean], {'True Location', 'Mean Estimate'}, 'Location', 'best');
    view(3);
    rotate3d on;


% Helper: DPD Cost Function
function J = dpd_cost(pos, r_received, Sensors, c, freqs)
    % pos: [x, y, z] hypothesis
    M = size(Sensors, 1);
    % size(M) = num of sensors

    % Calculate delays for hypothesis
    d_hyp = sqrt(sum((Sensors - pos).^2, 2));
    % size(d_hyp) = Mx1

    tau_hyp = d_hyp / c;
    % size(tau_hyp) = Mx1

    % Shift signals to align
    R_fft = fft(r_received);
    % size(R_fft) = N_samples x M
    Y_aligned = zeros(size(R_fft));
    % size(Y_aligned)= N_samples x M

    for k = 1:M
        % Phase shift +tau to align at t0
        phase = exp(1j * 2 * pi * freqs * tau_hyp(k));
        Y_aligned(:, k) = ifft(R_fft(:, k) .* phase);
    end
    
    % Correlation Matrix
    Q = Y_aligned' * Y_aligned;
    % size(Q) = MxM

    % Metric: Max Eigenvalue (Signal Energy)
    % We return negative because fminsearch minimizes
    J = max(eig(real(Q))); 
end

% Helper: Ellipsoid Plotter
function plot_confidence_ellipsoid(mu, Sigma, confidence)
    % mu: Mean vector [1x3]
    % Sigma: Covariance matrix [3x3]
    % confidence: Confidence level (e.g., 0.95)

    [V, D] = eig(Sigma);
    
    % Scale factor for 3D confidence ellipsoid
    % (x-mu)' * inv(Sigma) * (x-mu) <= chi2inv(conf, 3)
    chi_sq_val = chi2inv(confidence, 3); 
    radii = sqrt(diag(D) * chi_sq_val);
    
    % Generate unit sphere
    [xc, yc, zc] = sphere(30);
    
    % Transform sphere to ellipsoid
    % Start with coordinates [xc(:), yc(:), zc(:)]
    unit_sphere_pts = [xc(:) yc(:) zc(:)]';
    
    % Rotate and Scale
    ellipsoid_pts = V * (diag(radii) * unit_sphere_pts);
    
    % Shift to center
    X = reshape(ellipsoid_pts(1,:) + mu(1), size(xc));
    Y = reshape(ellipsoid_pts(2,:) + mu(2), size(yc));
    Z = reshape(ellipsoid_pts(3,:) + mu(3), size(zc));
    
    % Plot
    surf(X, Y, Z, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', 'r');
    plot3(X, Y, Z, 'r.', 'MarkerSize', 1); % Optional: Outline dots
end

% 4. Plotting (3D Figure 7)
    figure('Color','w', 'Position', [100, 100, 800, 600]);
    hold on; grid on; axis equal;
    
    % A0. Plot Sensor Locations (NEW CODE)
    h_sensors = scatter3(Sensors(:,1), Sensors(:,2), Sensors(:,3), 150, 'k', 's', 'filled');
    
    % A. Scatter Plot of Estimates
    % scatter3(Estimates(:,1), Estimates(:,2), Estimates(:,3), 15, 'b', 'filled');
    
    % B. Plot 95% Confidence Ellipsoid
    % plot_confidence_ellipsoid(Mean_Est, Cov_Est, 0.95);
    
    % C. Plot True Position (Large green marker)
    h_true = scatter3(Emitter_True(1), Emitter_True(2), Emitter_True(3), 200, 'g', 'h', 'filled');
    
    % D. Plot Mean Estimate (Large black marker)
    h_mean = scatter3(Mean_Est(1), Mean_Est(2), Mean_Est(3), 200, 'r', 'x', 'LineWidth', 3);
    
    xlabel('X (km)'); ylabel('Y (km)'); zlabel('Z (km)');
    title(sprintf('Physical placement of sensors (for M = 5) and emitter position used for simulation.\n(ASNR = %d dB, N = %d)', ASNR_dB, N_trials));
    
    % E. Update Legend to include Sensors
    legend([h_sensors, h_true, h_mean], {'Sensor Locations', 'True Location', 'Mean Estimate'}, 'Location', 'best');
    
    view(3);
    rotate3d on;
