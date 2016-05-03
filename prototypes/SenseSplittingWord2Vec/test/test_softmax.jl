using SoftmaxClassifier 
using FactCheck
import JLD


#data_dir = joinpath(Pkg.dir("Word2Vec"), "test", "data")
data_dir = joinpath("data") #For local run from testind directory
train_file = joinpath(data_dir, "mnist_train.jld")
test_file = joinpath(data_dir, "mnist_test.jld")

function test_softmax_training()
    println("Testing the softmax classifier on the MNIST dataset")

    println("Loading...")    
    X_train, y_train = JLD.load(train_file,"X_train", "y_train")
    X_test, y_test = JLD.load(test_file,"X_test", "y_test")

    println("Start training...")
    c = LinearClassifier(10, 784)
    acc= NaN
    for j in 1:5
        println("iteration $(j)")
        for i in 1:size(X_train,1)
            train_one!(c, X_train[i,:], y_train[i])
        end
        acc = accuracy(c, X_test, y_test)
        println("Accuracy on test set $acc")
    end
    c, acc
    
end


facts() do 
    classifier, final_accurasy=test_softmax_training()
    @fact final_accurasy --> greater_than(0.8)
        
    
end
    
    
