push!(LOAD_PATH,"../src/")
using Word2Vec
using FactCheck

#data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
data_dir = joinpath("data") #For local run from testind directory
train_file = joinpath(data_dir, "mnist_train.csv")
test_file = joinpath(data_dir, "mnist_test.csv")

function test_softmax()
    println("Testing the softmax classifier on the MNIST dataset")

    println("Loading...")
    D = readcsv(train_file, header=true)[1]
    X_train = D[:, 2:end] / 255
    y_train = map(Int64, D[:,1] + 1)
    

    D = readcsv(test_file, header=true)[1]
    X_test = D[:, 2:end] / 255
    if isless(Base.VERSION, v"0.4.0-")
        y_test = map(int64, D[:, 1] + 1)
    else
        y_test = map(Int64, D[:, 1] + 1)
    end

    println("Start training...")
    c = LinearClassifier(10, 784)
    acc= NaN
    for j in 1:5
        println("iteration $(j)")
        for i in 1:size(X_train,1)
            Word2Vec.train_one!(c, X_train[i,:], y_train[i])
        end
        acc = accuracy(c, X_test, y_test)
        println("Accuracy on test set $acc")
    end
    acc
    
end
facts() do 
    final_accurasy=test_softmax()
    @fact final_accurasy --> greater_than(0.8)
end