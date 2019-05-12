function result = apply_classifier(image, classifiers, boosted, ...
                                   face_vertical, face_horizontal, scales)
                               
% function result = apply_classifier(image, classifiers, boosted, ...
%                                    face_vertical, face_horizontal, scales)

result = ones(size(image)) * -10;
max_scales = zeros(size(image));

for scale = scales;
    % for efficiency, we either downsize the image, or the template, 
    % depending on the current scale
    scaled_image = imresize(image, 1/scale, 'bilinear');
    temp_result = apply_classifier(scaled_image, classifiers, boosted, ...
                                   face_vertical, face_horizontal, scales);
    temp_result = imresize(temp_result, size(image), 'nearest');
    
    higher_maxes = (temp_result > result);
    max_scales(higher_maxes) = scale;
    result(higher_maxes) = temp_result(higher_maxes);
end

