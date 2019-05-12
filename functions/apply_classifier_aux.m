function result = apply_classifier_aux(image, boosted, ...
                                       weak_classifiers, face_size)

% function result = apply_classifier_aux(image, boosted_classifiers, ...
%                                        weak_classifiers, face_size)
                                   

integral = integral_image(image);
face_vertical = face_size(1);
face_horizontal = face_size(2);

vertical_size = size(image, 1);
horizontal_size = size(image, 2);
result = zeros(vertical_size, horizontal_size);
classifier_number = size(boosted, 1);

for weak_classifier = 1:classifier_number
    classifier_index = boosted(weak_classifier, 1);
    classifier_alpha = boosted(weak_classifier, 2);
    classifier_threshold = boosted(weak_classifier, 3);
    classifier = weak_classifiers{classifier_index};
    
    for vertical = 1:(vertical_size-face_vertical+1)
        for horizontal = 1:(horizontal_size-face_horizontal+1)
            response1 = eval_weak_classifier(classifier, integral, vertical, horizontal);
            if (response1 > classifier_threshold)
                response2 = 1;
            else
                response2 = -1;
            end
            response = classifier_alpha * response2;
            row = vertical + round(face_vertical/2);
            col = horizontal + round(face_horizontal/2);
            result(row, col) = result(row, col) + response;
        end
    end
end