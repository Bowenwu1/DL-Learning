import load_data

test = load_data.unpickle("./dataset/cifar-10-batches-py/data_batch_1")
print(test.items())