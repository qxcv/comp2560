function show_unary(u)
    % SHOW_UNARY Visualise a CNN-returned unary as a heatmap
    probs = exp(u);
    normed = probs ./ sum(reshape(probs, [], 1));
    image(normed * 256);