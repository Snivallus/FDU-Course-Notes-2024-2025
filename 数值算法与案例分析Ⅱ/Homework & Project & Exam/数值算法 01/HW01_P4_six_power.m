% Set up parameters
MAXITR = 256; % Maximum number of iterations
COLMAG = 1; % Color magnitude (adjusts the color intensity over iterations)
ZROEPS = 1e-5; % Convergence threshold for Newton's method
IMGSIZ = 2048; % Image size

% Define the complex 6th roots of unity
r1 = 1;
r2 = exp(2*pi*1i/6);
r3 = exp(4*pi*1i/6);
r4 = exp(6*pi*1i/6);
r5 = exp(8*pi*1i/6);
r6 = exp(10*pi*1i/6);

% Create an empty image canvas
imageCanvas = zeros(IMGSIZ, IMGSIZ, 3, 'uint8'); % 3 channels for RGB image

% Set up the real and imaginary parts of the grid
[x, y] = meshgrid(linspace(-1, 1, IMGSIZ), linspace(-1, 1, IMGSIZ));

% Iterate over each pixel in the canvas
for row = 1:IMGSIZ
    for col = 1:IMGSIZ
        z = x(row, col) + 1i * y(row, col); % Convert to complex number
        for count = 1:MAXITR
            % Check convergence with roots
            if abs(z - r1) <= ZROEPS    % Red color
                imageCanvas(row, col, 1) = min(255, 255 - count * COLMAG); 
                break;
            elseif abs(z - r2) <= ZROEPS    % Green color
                imageCanvas(row, col, 2) = min(255, 255 - count * COLMAG); 
                break;
            elseif abs(z - r3) <= ZROEPS    % Blue color
                imageCanvas(row, col, 3) = min(255, 255 - count * COLMAG); 
                break;
            elseif abs(z - r4) <= ZROEPS    % Yellow color
                imageCanvas(row, col, 1) = min(255, 255 - count * COLMAG);
                imageCanvas(row, col, 2) = min(255, 255 - count * COLMAG);
                break;
            elseif abs(z - r5) <= ZROEPS    % Cyan color
                imageCanvas(row, col, 2) = min(255, 255 - count * COLMAG);
                imageCanvas(row, col, 3) = min(255, 255 - count * COLMAG); 
                break;
            elseif abs(z - r6) <= ZROEPS    % Magenta color
                imageCanvas(row, col, 3) = min(255, 255 - count * COLMAG);
                imageCanvas(row, col, 1) = min(255, 255 - count * COLMAG);
                break;
            elseif abs(z) <= ZROEPS
                break; % Stop if we hit the origin
            end
            % Newton's method update (finding 6th roots of unity)
            z = z - (z^6 - 1) / (6 * z^5); % Apply Newton's method for x^6 - 1 = 0
        end
    end
end

% Save the image to a file
imwrite(imageCanvas, 'newton_z6.png');

% Display the image
imshow(imageCanvas);
title('Convergence Image for x^6 - 1 = 0 using Newton''s Method');