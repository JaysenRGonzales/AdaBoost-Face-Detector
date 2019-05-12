function result = cascade_classify(window)

% If window is less than a predefined threshold, reject window

threshold = 1;

for i = 1:size(window)
    if window < threshold
        window = -1;
        result = window; 
        disp('Classifier ', i,  ' was rejected, it is not a face!');
    elseif window > threshold
        window = 1;
        result = window; 
        disp('Classifier ', i,  ' was classified as a possible face!');
    end
    
end

end

