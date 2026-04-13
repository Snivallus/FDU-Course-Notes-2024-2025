% Set up parameters
MAXITR = 128; % Maximum number of iterations
COLMAG = 10; % Color magnitude (this adjusts the color intensity over iterations)
ZROEPS = 1e-5; % Convergence threshold for Newton's method
IMGSIZ = 2048; % Image size

% Complex roots of x^3 = 1
r1 = 1;
r2 = exp(2*pi*1i/3);    % r2 = -0.5 + (sqrt(3)/2)*1i;
r3 = exp(4*pi*1i/3);    % r3 = -0.5 - (sqrt(3)/2)*1i;

% Create an empty image canvas
imageCanvas = zeros(IMGSIZ, IMGSIZ, 3, 'uint8'); % 3 channels for RGB image

% Set up the real and imaginary parts of the grid (canvas range [-2, 2])
[x, y] = meshgrid(linspace(-2, 2, IMGSIZ), linspace(-2, 2, IMGSIZ));

% Iterate over each pixel in the canvas
for row = 1:IMGSIZ
    for col = 1:IMGSIZ
        z = x(row, col) + 1i * y(row, col); % Convert to complex number
        for count = 1:MAXITR
            % Check convergence with roots
            if abs(z - r1) <= ZROEPS
                imageCanvas(row, col, 1) = min(255, 255 - count * COLMAG); % Red color
                break;
            elseif abs(z - r2) <= ZROEPS
                imageCanvas(row, col, 2) = min(255, 255 - count * COLMAG); % Green color
                break;
            elseif abs(z - r3) <= ZROEPS
                imageCanvas(row, col, 3) = min(255, 255 - count * COLMAG); % Blue color
                break;
            elseif abs(z) <= ZROEPS
                break; % Stop if we hit the origin
            end
            % Newton's method update (finding cube roots of unity)
            z = z - (z^3 - 1) / (3 * z^2);
        end
    end
end

% Save the image to a file
imwrite(imageCanvas, 'newton_fractal_x^3=1.png');

% Display the image
imshow(imageCanvas);
title('Convergence Image using Newton''s Method');
