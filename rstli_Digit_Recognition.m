% Load the USPS dataset
load('USPS.mat'); 

% Set up a figure for displaying the images
figure;
% the number of rows and columns of train_patterns
[size_rows, size_columns] = size(train_patterns);

% Display the size
disp(['The size of train_patterns is ' num2str(size_rows) ' x ' num2str(size_columns)]);

% Loop through the first 16 images
for i = 1:16
    % Select the i-th pattern (column)
    pattern = train_patterns(:,i);
    
    % Reshape the pattern into a 16x16 matrix
    % Note: MATLAB stores images in column-major format,
    % so we actually don't need to transpose the image after reshaping.
    imageMatrix = reshape(pattern, 16, 16);
    
    % Display the image in a 4x4 grid
    subplot(4, 4, i);
    imagesc(imageMatrix); % Display the image
    colormap(gray); % Use a grayscale colormap
    axis image; % Ensure the image isn't distorted
    axis off; % Hide the axes for clarity
end

% save this figure to a file
saveas(gcf, 'digit_images.png');

%Step2 start

% There are ten digits (0-9)
num_digits = 10; 
%initialize train averages
train_aves = zeros(256, num_digits); % Initialize the matrix to store mean digits

for k = 1:num_digits
    % Extract all instances of digit k-1
    digit_patterns = train_patterns(:, train_labels(k, :) == 1);
    % Compute the mean image for digit k-1 and store it
    train_aves(:, k) = mean(digit_patterns, 2);
end
disp("The train aves");
disp(train_aves);

% Display mean digits
figure;
for k = 1:num_digits
    subplot(2, 5, k);
    imagesc(reshape(train_aves(:, k), 16, 16));
    colormap(gray);
    title(['Digit ', num2str(k-1)]);
    axis image;
    axis off;
end

% Save the figure as a PDF
saveas(gcf, 'mean_digits.pdf');

%Step 3 starts

% Step3 part a
% Initialize the classification matrix
test_classif = zeros(10, 4649); 

for k = 1:10
    % Compute the squared Euclidean distances for each test image from the k-th mean digit
    test_classif(k, :) = sum((test_patterns - repmat(train_aves(:, k), [1, 4649])).^2);
end

% Step3 part b
test_classif_res = zeros(1, 4649);
for j = 1:4649
    [~, ind] = min(test_classif(:, j));
    test_classif_res(j) = ind;
end
disp("Test classification result");
disp(test_classif_res);
% Step3 part c
test_confusion = zeros(10, 10); % Initialize the confusion matrix

for k = 1:10
    % Extract classification results for test images of true digit k-1
    tmp = test_classif_res(test_labels(k, :) == 1);
    
    % Count occurrences of each classification result
    for j = 1:10
        test_confusion(k, j) = sum(tmp == j);
    end
end

disp("Test confusion result of part 3");
% Display the confusion matrix
disp(test_confusion);

%part 4 begin
%part 4 a Compute rank 17 SVD for each digit
% Initialize the array to store the left singular vectors
train_u = zeros(256, 17, 10); 

for k = 1:10
    % Compute the rank 17 SVD for the k-th digit
    [train_u(:, :, k), ~, ~] = svds(train_patterns(:, train_labels(k, :) == 1), 17);
end
%part 4 b
test_svd17 = zeros(17, 4649, 10); % Initialize the array for expansion coefficients

for k = 1:10
    % Compute the coefficients for each test image and the k-th digit's singular vectors
    test_svd17(:, :, k) = train_u(:, :, k)' * test_patterns;
end
%part 4 c Compute approximation errors
 % Initialize the matrix to store approximation errors
test_svd17res = zeros(10, 4649);

for k = 1:10
    % Calculate the rank 17 approximation for each test image under the k-th digit's singular vectors
    approximations = train_u(:, :, k) * test_svd17(:, :, k);
    % Compute and store the squared error for these approximations
    test_svd17res(k, :) = sum((test_patterns - approximations) .^ 2);
end

% part 4 d  Compute confusion matrix
% Initialize the confusion matrix of test svd17
test_svd17_confusion = zeros(10, 10);

% Classify each test image
[test_min, test_svd17_classif_res] = min(test_svd17res, [], 1);

for k = 1:10
    % Extract classification results for test images of true digit k-1
    true_labels = find(test_labels(k, :) == 1);
    classified_as = test_svd17_classif_res(true_labels);
    
    % Count occurrences of each classification result
    for j = 1:10
        test_svd17_confusion(k, j) = sum(classified_as == j);
    end
end

disp("Test confusion result of part 4");
% Display the confusion matrix
disp(test_svd17_confusion);
