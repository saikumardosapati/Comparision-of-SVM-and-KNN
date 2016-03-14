function [weight,bias, truePositive, falsePositive,alp,tn,fn]=smo(test,testclasses,train,trainclasses, kernel, eps)

    posCount = 0;
    negCount = 0;
    trainrowCount = size(train,1);
    testrowCount = size(test,1);
    attributeCount = size(test,2);
    for i = 1:trainrowCount
        if trainclasses(i) == 1
            posCount = posCount + 1;
        else
            negCount = negCount + 1;
        end
    end

    alpha = randi([0,1000], [trainrowCount, 1]);
    diff = dot(alpha, trainclasses);

    for i = 1:posCount
        alpha(i) = alpha(i) - (diff / posCount);
    end

    b = zeros(trainrowCount, 1);
    
    % limit the number of
    % iterations
    for loop = 1:400
       
        % calculate the weight vector
        w = zeros(1, attributeCount);
        for i = 1:trainrowCount
            w = w + alpha(i) * trainclasses(i) * train(i, :);
        end

        % calculate KKT
        kkt = zeros(trainrowCount, 1);
        for i = 1:trainrowCount
            kkt(i) = alpha(i) * (trainclasses(i) * (dot(w, train(i, :)) + b(i)) - 1);
        end

        % find the index of the max 
        [dummy, maxKKT] = max(kkt);
    %     maxKKT = randi([1, rowCount]);

        % calculate e
        e = zeros(trainrowCount,1);
        for i = 1:trainrowCount    
            for j = 1:trainrowCount
                e(i) = e(i) +  alpha(j) * trainclasses(j) * (kernel(j, maxKKT) - kernel(j, i)) + trainclasses(i) - trainclasses(maxKKT);
            end
        end

        [dummy, maxE] = max(abs(e));
   

        k = kernel(maxKKT, maxKKT) + kernel(maxE, maxE) - 2 * kernel(maxKKT, maxE);

        oldAlphaMaxE = alpha(maxE);
        alpha(maxE) = oldAlphaMaxE + (trainclasses(maxE) * e(maxE)) / k;

        % clip negative alpha values
        if alpha(maxE) < 0
            alpha(maxE) = 0;
        end

        oldAlphaMaxKKT = alpha(maxKKT);
        alpha(maxKKT) = oldAlphaMaxKKT + trainclasses(maxE) * trainclasses(maxKKT) * (oldAlphaMaxE - alpha(maxE));

        if alpha(maxKKT) < 0
            alpha(maxKKT) = 0;
        end

        for i = 1:trainrowCount
            if alpha(i) > 0
                b(i) = trainclasses(i) - dot(w, train(i, :));
            end
        end

        % calculate the final bias
        biasCount = 0;
        for i = 1:trainrowCount
            if alpha(i) ~= 0
                biasCount = biasCount + 1;
            end
        end
        totalB = sum(b) / biasCount;

        % calculate the weight vector
        w = zeros(1, attributeCount);
        for i = 1:trainrowCount
            w = w + alpha(i) * trainclasses(i) * train(i, :);
        end

        truePositiveCount = 0;
        falsePositiveCount = 0;
        trueNegativeCount = 0;
        falseNegativeCount = 0;
        
        for i = 1:testrowCount
            output = sign(dot(w, test(i, :)) + totalB);
            if output == 1
               if output == testclasses(i)
                   truePositiveCount = truePositiveCount + 1;
               else
                   falsePositiveCount = falsePositiveCount + 1;
               end
            else
                if output == testclasses(i)
                    trueNegativeCount = trueNegativeCount + 1;
                else
                    falseNegativeCount = falseNegativeCount + 1;
                end
            end
        end
        
        truePositive = truePositiveCount; %/ (truePositiveCount + falseNegativeCount);
        falsePositive = falsePositiveCount;% / (falsePositiveCount + falseNegativeCount);
        tn=trueNegativeCount;
        fn=falseNegativeCount;
        % break if we reached the desired accuracy (based on the param eps)
        if 1 - (truePositiveCount + trueNegativeCount) / (truePositiveCount + trueNegativeCount + falseNegativeCount + falsePositiveCount) < eps
            break
        end
    end
    weight=w;
    bias=b;
    alp=alpha;
end
