using DataFrames
using Images
using DecisionTree

IMAGE_SIZE = 20 * 20

function read_bmp(data_frame::DataFrame, path::AbstractString)
    x = zeros(size(data_frame, 1), IMAGE_SIZE)
    for (index, ID) in enumerate(data_frame[:, :ID])
        img = load("$(path)/$ID.Bmp")
        
        if img != nothing
            gray_image = float(convert(Image{Colors.Gray}, img))
            x[index, :] = reshape(gray_image, 1, IMAGE_SIZE).data
        end
    end
    return x
end

function read_train_and_test()
    label_df = readtable("trainLabels.csv")
    test_df = readtable("sampleSubmission.csv")

    yTrain = convert(Array{Int, 1}, map(x -> x[1], label_df[:, :Class]))
    
    xTrain = read_bmp(label_df, "trainResized")
    xTest = read_bmp(test_df, "testResized")
    
    train_df = DataFrame(xTrain)
    names!(train_df, [symbol("X$x") for x in 1:IMAGE_SIZE])

    test_df = DataFrame(xTest)
    names!(test_df, [symbol("X$x") for x in 1:IMAGE_SIZE])
    
    train_df[:Class] = yTrain

    train_df, test_df
end

function random_forest()
    (train, test) = read_train_and_test()
    
    # model = build_forest(yTrain, xTrain, 20, 50, 1.0)
    # predTest = apply_forest(model, xTest)

    # yTest[:, :Class] = map(Char, predTest)
    # writetable("juliaSubmission.csv", yTest, separator=',', header=true)
    
    x = Matrix(train[:, [symbol("X$x") for x in 1:IMAGE_SIZE]])
    y = convert(Array{Int, 1}, train[:,:Class])
    
    accuracy = nfoldCV_forest(y, x, 20, 50, 2, 1.0)
    println(mean(accuracy))
end

function add_random_selection_column!(data_frame)
    data_frame[:SelectionKey] = rand(nrow(data_frame))
end

function selected_for_test(selection_key, split_index, split_count)
    range = 1 / split_count
    lower_bound = range * split_index
    upper_bound = range * split_index + range
    return lower_bound <= selection_key && selection_key < upper_bound
end


function nfoldCV(split::Int, func)
    (train, test) = read_train_and_test()

    # To accelerate development, use 30% subset of the data for now.
    subset = rand(nrow(train))
    train = train[subset .< 0.3, :]

    println(hist(train[:Class]))
    
    add_random_selection_column!(train)
    selection_key = train[:, :SelectionKey]
    
    for split_index in 0:split-1
        selected_rows::Vector{Bool} = map(key->selected_for_test(key, split_index, split), selection_key)
        
        x = Matrix(train[!selected_rows, [symbol("X$x") for x in 1:IMAGE_SIZE]])
        y = convert(Array{Int, 1}, train[!selected_rows, :Class])

        x_test = Matrix(train[selected_rows, [symbol("X$x") for x in 1:IMAGE_SIZE]])
        y_test = convert(Array{Int, 1}, train[selected_rows, :Class])

        predTest = func(y, x, x_test)

        println(sum(y_test .== predTest) / length(predTest))
    end

end

function my_random_forest(train_y, train_x, test_x)
    model = build_forest(train_y, train_x, 20, 50, 1.0)
    apply_forest(model, test_x)
end

function my_knn(train_y, train_x, test_x)
    k = 1
    function find_k_nearest_index(k, test_record, train_x)
        distance_list = Array(Float64, size(train_x, 1))
        for train_row_index in 1:size(train_x, 1)
            train_record = train_x[train_row_index, :]

            distance_list[train_row_index] = sum((test_record - train_record) .^ 2)
        end

        nearest_k_index = sortperm(distance_list)[1:k]
        
        # Remove distance > shortest_distance * 5
        nearest_k_index = filter(x -> distance_list[x] <= distance_list[nearest_k_index[1]] * 2, nearest_k_index)
#        println(length(nearest_k_index))
        
        return nearest_k_index
    end

    function get_most_popular_label(labels::Array{Int, 1})
        dict = Dict{Int, Int}(map(label->(label, 0), labels))
        highest_count = 0
        most_popular_label = 0
        for label in labels
            dict[label] += 1
            if dict[label] > highest_count
                highest_count = dict[label]
                most_popular_label = label
            end
        end
        most_popular_label
    end
    
    prediction_index = Array(Any, size(test_x, 1))
    
    prediction = zeros(size(test_x, 1))
    for test_row_index in 1:size(test_x, 1)
        test_record = test_x[test_row_index, :]
#        prediction_index[test_row_index] = @spawn find_k_nearest_index(k, test_record, train_x)
        prediction_index[test_row_index] = find_k_nearest_index(k, test_record, train_x)
    end        

    for test_row_index in 1:size(test_x, 1)
        predicted_labels = train_y[fetch(prediction_index[test_row_index])]
        prediction[test_row_index] = get_most_popular_label(predicted_labels)
    end
    prediction
end

# @time random_forest()

# nfoldCV(2, my_random_forest)

#nfoldCV(5, my_knn)
nfoldCV(5, my_random_forest)

