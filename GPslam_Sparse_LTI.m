%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Continuous-time trajectory estimation with landmarks
% Code by Shunmei Dong
% Email: shunmei.dong@bit.edu.cn
%
% Ref: Anderson, Sean, et al. "Batch nonlinear continuous-time trajectory 
% estimation as exactly sparse Gaussian process regression." Autonomous 
% Robots 39 (2015): 221-238.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; 
close all; 
clc;
rng(0)

%% Parameters
T = 20;              % Total duration
dt = 0.5;            % Sampling interval
time = 0:dt:T;       
N = length(time);

%% Robot True Trajectory (Constant Velocity)
v_x_true = 1;        % m/s
v_y_true = 0.5;      % m/s
true_x   = v_x_true * time;  % True position of the robot
true_y   = v_y_true * time;  

%% Noisy Robot Measurements
sigma_meas = 0.1;   % Measurement noise standard deviation
meas_x     = true_x + sigma_meas*randn(1,N);  % Noisy position measurements
meas_y     = true_y + sigma_meas*randn(1,N);  

%% Landmarks
L = 10;   % Number of landmarks
range_x = [-5, T*v_x_true+5];  % range for landmark generation
range_y = [-5, T*v_y_true+5]; 
landmark_x = (range_x(2)-range_x(1)) * rand(L,1) + range_x(1);  % Random positions of landmarks
landmark_y = (range_y(2)-range_y(1)) * rand(L,1) + range_y(1);  
landmarks_true = [landmark_x, landmark_y];  

%% GP Prior
F = [0 0 1 0;   % State transition matrix
     0 0 0 1;
     0 0 0 0;
     0 0 0 0];
F_d = eye(4) + F * dt;  % Discrete-time approximation

Q_c = 0.1 * eye(2);  % Process noise covariance
Q_d = zeros(4,4);    % Discretized process noise covariance
Q_d(1:2,1:2) = (dt^3/3) * Q_c; 
Q_d(1:2,3:4) = (dt^2/2) * Q_c; 
Q_d(3:4,1:2) = (dt^2/2) * Q_c; 
Q_d(3:4,3:4) = (dt)     * Q_c; 
inv_Q_d = inv(Q_d);  % Inverse of the process noise covariance

%% The Inverse of the Kernel Matrix
F_d_matrix = sparse(4*N, 4*N);  
for i = 1:N
    idx = 4*(i-1) + (1:4);
    F_d_matrix(idx, idx) = speye(4);
    for j = i+1:N
        idx_next = 4*(j-1) + (1:4);
        F_d_matrix(idx_next, idx) = F_d^(j-i);
    end
end
F_d_matrix = tril(F_d_matrix);  % Make the matrix lower triangular

big_inv_Q = kron(speye(N), inv(Q_d));
P_inv_robot = inv(F_d_matrix') * big_inv_Q * inv(F_d_matrix);  % The inverse of the kernel matrix

%% Prior Mean function
X_mean = zeros(4, N);  % Robot state prior mean (position and velocity)
X_mean(:,1) = [0; 0; 1; 0.5];  % Initial prior: position [0,0], velocity [1, 0.5]
for k = 2:N
    X_mean(:,k) = F_d * X_mean(:,k-1);  % Propagate the prior state using the transition model
end
X_mean_vec = reshape(X_mean, 4*N, 1);  % Flatten robot prior mean into a vector

%% Landmarks Prior
L_mean = landmarks_true;  % Landmark prior mean
P_landmark = 100 * eye(2*L);  % Landmark prior covariance
P_inv_landmark = inv(P_landmark);  % Inverse of the landmark prior covariance
L_mean_vec = reshape(L_mean', 2*L, 1);


%% Combined State: z = [robot_states (4N x 1); landmarks (2L x 1)]
z0 = [X_mean_vec; L_mean_vec];

%% Measurement Models
R_robot = sigma_meas^2 * eye(2);  % Robot measurement noise covariance
R_robot_inv = inv(R_robot);       
H_robot = [1 0 0 0;               % Measurement matrix for robot position
           0 1 0 0];              

% Jacobian matrix for robot measurements
G_robot = sparse(2*N, 4*N);
for i = 1:N
    row_idx = 2*(i-1) + (1:2);
    col_idx = 4*(i-1) + (1:4);
    G_robot(row_idx, col_idx) = H_robot;
end

sigma_range = 0.1;  % Standard deviation for range measurements
sigma_bearing = 0.05;  % Standard deviation for bearing measurements
R_landmark = blkdiag(kron(eye(L), sigma_range^2), kron(eye(L), sigma_bearing^2));  % Landmark measurement noise covariance
R_landmark_inv = inv(R_landmark);  % Inverse of the landmark measurement noise covariance

%% Combine Measurement Data
Y_robot = [meas_x; meas_y];  % Combine robot position measurements
Y_robot_vec = reshape(Y_robot, 2*N, 1);

% Generate and combine landmark measurements
Y_landmark = zeros(2*L, N);
for n = 1:N
    x_r = true_x(n);  % True position of the robot at time n
    y_r = true_y(n); 
    for i = 1:L
        dx = landmarks_true(i,1) - x_r;  % Difference in position between robot and landmark
        dy = landmarks_true(i,2) - y_r;  
        range_meas = sqrt(dx^2 + dy^2) + sigma_range*randn;  % Range measurement with noise
        bearing_meas = atan2(dy, dx) + sigma_bearing*randn;  % Bearing measurement with noise
        Y_landmark( (i-1)*2 + 1, n) = range_meas;  % Store range measurement
        Y_landmark( (i-1)*2 + 2, n) = bearing_meas;  % Store bearing measurement
    end
end
Y_landmark_vec = reshape(Y_landmark, 2*L*N, 1);

Y_vec = [Y_robot_vec; Y_landmark_vec];  % Combine all measurements (robot + landmarks)

%% Combined the inverse of the Kernel Matrix for the whole state z
P_inv_comb = blkdiag(P_inv_robot, P_inv_landmark);  % Combine robot and landmark prior precision matrices

%% Jacobian Matrix G for Combined Measurement
G_robot_comb = [G_robot, sparse(2*N, 2*L)];

G_landmark = sparse(2*L*N, 4*N + 2*L);
for n = 1:N
    idx_r = 4*(n-1) + (1:4);  % Robot state indices at time n
    x_r0 = X_mean(1,n);  % position of robot at time n
    y_r0 = X_mean(2,n);
    for i = 1:L
        idx_l = 4*N + 2*(i-1) + (1:2);  % Landmark state indices
        dx = L_mean(i,1) - x_r0;        % difference between robot and landmark
        dy = L_mean(i,2) - y_r0;        
        range_pred = sqrt(dx^2 + dy^2); % Predicted range between robot and landmark
        if range_pred < 1e-6
            range_pred = 1e-6;          % Avoid division by zero if range is too small
        end
        % Compute Jacobian for robot position with respect to range and bearing
        d_range_dx = -dx / range_pred;
        d_range_dy = -dy / range_pred;
        d_bearing_dx = dy/(range_pred^2);
        d_bearing_dy = -dx/(range_pred^2);
        J_r = [d_range_dx, d_range_dy;
               d_bearing_dx, d_bearing_dy];  % Jacobian for robot
        J_l = -J_r;                          % Jacobian for landmark (negative of robot's)

        row_idx = 2*((n-1)*L + (i-1)) + (1:2);  % Row indices for the measurement matrix
        G_landmark(row_idx, idx_r(1:2)) = J_r;  % Fill the Jacobian for robot position
        G_landmark(row_idx, idx_l) = J_l;  % Fill the Jacobian for landmark position
    end
end

% Final combined Jacobian matrix
G = [G_robot_comb;
     G_landmark];

R_inv_robot_big = kron(speye(N), R_robot_inv);  
R_inv_landmark_big = kron(speye(N), R_landmark_inv);  
R_inv_comb = blkdiag(R_inv_robot_big, R_inv_landmark_big);  % Combined inverse covariance matrix

%% Optimization: Gauss-Newton Iteration for Combined State
max_iter = 100;  % Maximum number of iterations
z_op = zeros(4*N+2*L,1);  % Initial guess for the state offset
for iter = 1:max_iter
    z_curr = z0 + z_op;  % Current estimate of the combined state
    x_curr = z_curr(1:4*N);  % Robot states (position and velocity)
    L_curr = z_curr(4*N+1:end);  % Landmarks positions

    h_robot = G_robot * x_curr;  % Robot measurement prediction
    h_landmark = zeros(2*L*N,1);  % Initialize landmark measurement prediction
    for n = 1:N
        idx_r = 4*(n-1)+(1:4);  % Indexes for robot state at time n
        x_r = x_curr(idx_r);  % Current robot state
        x_r_pos = x_r(1);  % Robot x position
        y_r_pos = x_r(2);  % Robot y position
        for i = 1:L
            idx_l = 2*(i-1)+(1:2);  % Indexes for landmark i
            l_i = L_curr(idx_l);  % Current position of landmark i
            dx = l_i(1) - x_r_pos;  % Difference in x between robot and landmark
            dy = l_i(2) - y_r_pos;  % Difference in y between robot and landmark
            range_pred = sqrt(dx^2+dy^2);  % Predicted range between robot and landmark
            bearing_pred = atan2(dy, dx);  % Predicted bearing between robot and landmark
            row_idx = 2*((n-1)*L + (i-1)) + (1:2);  % Index for the measurement row
            h_landmark(row_idx) = range_pred;  % Store range prediction
            h_landmark(row_idx(end)) = bearing_pred;  % Store bearing prediction
        end
    end
    h = [h_robot; h_landmark];  % Combine robot and landmark predictions
    
    r = Y_vec - h;  % Residual
    lhs = P_inv_comb + G' * R_inv_comb * G;  % Left-hand side of the system
    rhs = -P_inv_comb * z_op + G' * R_inv_comb * r;  % Right-hand side of the system

    delta_z = lhs \ rhs;  % Solve for the update in the state
    z_op = z_op + delta_z;  % Update the state offset

    if norm(delta_z) < 1e-6  % Convergence check
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
end

z_hat = z0 + z_op;  % Final estimated state
x_hat_vec = z_hat(1:4*N);  % Estimated robot states
X_hat_full = reshape(x_hat_vec, 4, N)';
L_hat_vec = z_hat(4*N+1:end);  % Estimated landmarks
Landmarks_hat = reshape(L_hat_vec, 2, L)'; 

%% Posterior Covariance Calculation
H = lhs;  % Posterior information matrix
Sigma_full = inv(H);  % Posterior covariance matrix (inverse of information matrix)

%% Query the Trajectory for Robot States (GP Interpolation)
t_query = linspace(0, T, 50);  % Query times for interpolation
x_query = zeros(length(t_query), 4);
cov_query = zeros(length(t_query), 4, 4);

for iq = 1:length(t_query)
    tau = t_query(iq);  % Query time
    % Find the nearest time slot [t_n, t_{n+1}]
    idx_n = find(time <= tau, 1, 'last');  % Nearest time index before tau
    if idx_n >= N
        idx_n = N-1;  % Prevent out-of-bounds
    end
    idx_nplus = idx_n + 1;  % Next time index
    dt_frac = tau - time(idx_n);  % Fractional time difference

    % Calculate the prior mean at time tau
    Phi_partial = eye(4) + F * dt_frac;
    x_check_tau = Phi_partial * X_mean(:, idx_n);
    
    % Compute Lambda and Psi for the interpolation
    ratio = dt_frac / dt;
    Q_tau = ratio * Q_d;
    Phi_nplus_tau = eye(4) + F * (dt - dt_frac);
    Psi_tau = Q_tau * Phi_nplus_tau' * inv_Q_d;
    Lambda_tau = Phi_partial - Psi_tau * F_d;
    
    % Use the deviations at the two adjacent times
    xhat_n = X_hat_full(idx_n, :)';  % Estimated state at time n
    xmean_n = X_mean(:, idx_n);  % Prior mean state at time n
    dx_n = xhat_n - xmean_n;  % Deviation at time n
    
    xhat_nplus = X_hat_full(idx_nplus, :)';  % Estimated state at time n+1
    xmean_nplus = X_mean(:, idx_nplus);  % Prior mean state at time n+1
    dx_nplus = xhat_nplus - xmean_nplus;  % Deviation at time n+1
    
    % Compute the correlation term for the trajectory at time tau
    corrTerm = [Lambda_tau, Psi_tau] * [dx_n; dx_nplus];
    x_hat_tau = x_check_tau + corrTerm;  % Interpolated state at tau
    x_query(iq,:) = x_hat_tau';  % Store the interpolated state

    % Compute the covariance matrix at time tau
    Cov_tau = Sigma_full(4*(idx_n-1)+(1:4), 4*(idx_n-1)+(1:4));  % Covariance at tau
    cov_query(iq, :, :) = Cov_tau;  % Store the covariance matrix
end

%% Visualization
figure; hold on; grid on; axis equal;
% Plot the robot's prior trajectory
plot(X_mean(1,:), X_mean(2,:), 'bpentagram','MarkerSize',4, 'LineWidth',1);
% Plot the robot's position measurements
plot(meas_x,  meas_y,  'rx','MarkerSize',6, 'LineWidth',1.2);
% Plot the robot's estimated trajectory
plot(X_hat_full(:,1), X_hat_full(:,2), 'go','MarkerSize',4, 'LineWidth',1.2);
% Plot the interpolated query trajectory
plot(x_query(:,1), x_query(:,2), 'm*','MarkerSize',5, 'LineWidth',0.5);
% Plot the true landmarks
plot(landmarks_true(:,1), landmarks_true(:,2), 'ks','MarkerSize',8, 'LineWidth',2);
% Plot the estimated landmarks
plot(Landmarks_hat(:,1), Landmarks_hat(:,2), 'co','MarkerSize',8, 'LineWidth',2);

legend('Robot Prior','Robot Meas.','Robot Estimate','Robot Query','True Landmarks','Estimated Landmarks');
xlabel('X'); ylabel('Y');
title('Continuous-time Trajectory Estimation with Landmarks');

%% Plot Error Ellipses for Query Trajectory Points (2σ)
% Plot the error ellipses for the estimated points
for n = 1:N
        idx_xy = 4*(n-1) + (1:2);  % Indexes for the x and y positions of the robot
        Cov_xy = Sigma_full(idx_xy, idx_xy);  % Covariance matrix for robot position
        mu_xy  = X_hat_full(n,1:2);  % Estimated position of the robot
        draw_ellipse(mu_xy, Cov_xy, 2, 'r', 0.07);  % 2σ ellipse with low transparency
end

% Plot the error ellipses for the query points
for iq = 1:length(t_query)
    mu_query = x_query(iq, :);  % Query point position
    Cov_query = squeeze(cov_query(iq, :, :));  % Covariance of the query point
    draw_ellipse(mu_query, Cov_query(1:2, 1:2), 2, 'r', 0.2);
end

% Plot the error ellipses for the landmarks
for i = 1:L
    idx_l = 4*N + 2*(i-1) + (1:2);  % Indexes for the landmark positions
    Cov_l = Sigma_full(idx_l, idx_l);  % Covariance matrix for landmark
    mu_l  = Landmarks_hat(i,:);  % Estimated landmark position
    draw_ellipse(mu_l, Cov_l, 2, 'b', 0.2);
end


%% Draw Ellipse Function
function draw_ellipse(mu, Sigma, nSigma, colorStr, alphaVal)
    if nargin < 3, nSigma = 1; end  % Default number of sigma for ellipse
    if nargin < 4, colorStr = 'b'; end  % Default color of the ellipse
    if nargin < 5, alphaVal = 0.2; end  % Default transparency of the ellipse

    Sigma = (Sigma + Sigma')/2;
    Sigma = full(Sigma);

    [U,S,~] = svd(Sigma);  % Perform Singular Value Decomposition on the covariance matrix

    angles = linspace(0, 2*pi, 100);  % Create angles for the ellipse
    circle = [cos(angles); sin(angles)];  % Parametrize the unit circle

    ellipse = U * sqrt(S) * circle * nSigma;  % Scale and rotate the unit circle to form the ellipse
    X_ellipse = mu(1) + ellipse(1,:);  % X coordinates of the ellipse
    Y_ellipse = mu(2) + ellipse(2,:);  % Y coordinates of the ellipse

    fill(X_ellipse, Y_ellipse, colorStr, ...  % Plot the ellipse
         'FaceAlpha', alphaVal, ...  
         'EdgeColor', colorStr, ...  
         'LineWidth', 1.0,'HandleVisibility','off');
end
