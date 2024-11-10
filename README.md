# PCA-of-image
% Load an example grayscale image
original_image = imread('cameraman.tif'); % Replace 'cameraman.tif' with your image file
imshow(original_image), title('Original Image');
pause(1);

% Convert image to double for calculations
img = double(original_image);

% Reshape the image into a 2D matrix where each row represents a pixel
% For an MxN image, reshape to M x N matrix where each column is a pixel feature
[m, n] = size(img);
X = reshape(img, m, n);

% Step 1: Mean centering
X_mean = mean(X, 2); % Compute mean along each row
X_centered = X - X_mean;

% Step 2: Compute covariance matrix
covariance_matrix = (X_centered * X_centered') / (n - 1);

% Step 3: Calculate eigenvalues and eigenvectors
[eigenvectors, eigenvalues_matrix] = eig(covariance_matrix);
eigenvalues = diag(eigenvalues_matrix); % Extract eigenvalues from the diagonal

% Step 4: Sort eigenvalues and corresponding eigenvectors in descending order
[eigenvalues_sorted, idx] = sort(eigenvalues, 'descend');
eigenvectors_sorted = eigenvectors(:, idx);

% Select the top 3 principal components
num_components = 3;
selected_eigenvectors = eigenvectors_sorted(:, 1:num_components);

% Step 5: Project the image onto the top 3 principal components
X_reduced = selected_eigenvectors' * X_centered;

% Step 6: Visualize each of the top 3 components individually
figure;
for i = 1:num_components
    % Reconstruct the image using only the i-th principal component
    component_i = selected_eigenvectors(:, i) * X_reduced(i, :);
    component_image = reshape(component_i, m, n); % Reshape to original dimensions
    
    % Display the i-th principal component as an image
    subplot(1, num_components, i);
    imshow(uint8(component_image)), title(['Principal Component ', num2str(i)]);
end

% Display the original and reconstructed image for comparison
figure;
reconstructed_image = selected_eigenvectors * X_reduced + X_mean;
compressed_image = reshape(reconstructed_image, m, n);
subplot(1, 2, 1), imshow(uint8(original_image)), title('Original Image');
subplot(1, 2, 2), imshow(uint8(compressed_image)), title('Reconstructed Image (Top 3 Components)');
