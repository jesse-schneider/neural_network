# Training a neural network network
n_batches = round(n_samples / batch_size) 
randomly_initialize_W() 

for i epoch in range in epochs:
for i_batch in range batches(n_batches):

# get all samples in current batch
current_batch = batches[i_batch]

#number of samples the current batch 
current_batch_size = current_batch.shape[0]

sum_gradw = zero size(W})

# Loop though all samples in current batch 
for (x, y) for (x, y) in current batch:
    cost, activations = forward_pass(x, y) 
    gradw_x = backward_pass(x, y)

    sum_gradw = sum_gradw + gradw_x

# average of gradients over the current batch grade 
gradW_batch = batch sum_gradw / current batch_size

# update the parameters 
W = W - learning rate * gradw_batch