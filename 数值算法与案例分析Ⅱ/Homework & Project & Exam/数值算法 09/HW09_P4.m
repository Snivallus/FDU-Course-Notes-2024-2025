% Define the original quadrature points
orig_points = [2/3, 1/6;
               1/6, 2/3;
               1/6, 1/6];

% Define the affine transformations for each sub-triangle
% Each transformation now returns a single vector output
transformations = {
    @(p) [0.5*p(1) + 0.5*p(2), 0.5*p(2)];           % Sub1
    @(p) [0.5*p(1) + 0.5, 0.5*p(2)];                % Sub2
    @(p) [0.5*p(1), 0.5*p(2) + 0.5];                % Sub3
    @(p) [0.5*p(2), 0.5*p(1) + 0.5*p(2)];           % Sub4
};

% Initialize the composite estimate
composite_estimate = 0;

% Evaluate and sum contributions from all sub-triangles
for i = 1:4
    T = transformations{i}; % Current transformation
    for j = 1:3
        % Apply transformation to the j-th quadrature point
        p = orig_points(j, :);
        transformed_point = T(p);
        x_trans = transformed_point(1);
        y_trans = transformed_point(2);
        
        % Evaluate the function at the transformed point
        f_val = exp(x_trans) * sin(y_trans);
        
        % Accumulate the result with weight 1/24
        composite_estimate = composite_estimate + f_val / 24;
    end
end

% Compute the exact integral for comparison
exact_integral = 0.5*(exp(1) - sin(1) + cos(1)) - 1;

% Display results
fprintf('Composite Estimate: %.6f\n', composite_estimate);
fprintf('Exact Value: %.6f\n', exact_integral);