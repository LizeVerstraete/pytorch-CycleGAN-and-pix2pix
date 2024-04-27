import time
import random
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from agents.helpers.Evaluator import General_Evaluator
from results.traintestval_functions import train_loop, val_loop

if __name__ == '__main__':
    #SET VARIABLES
    data_folder = r"/esat/biomeddata/kkontras/r0786880/biopsy_data_filtered_aligned_small_412"
    print_intermediate_results = False
    save_dir = r"/esat/biomeddata/kkontras/r0786880/results/evaluation_some_training_small_tiles"

    #DON'T EDIT ANYTHING BELOW
    opt = TrainOptions().parse()   # get training options
    opt.aligned = True
    dataset_train = create_dataset(opt,data_folder)  # create a dataset_aligned given opt.dataset_mode and other options
    dataset_size_train = len(dataset_train)    # get the number of images in the dataset_aligned.
    print('The number of training images = %d' % dataset_size_train)

    opt.isTrain = False #set Train and Test to false to create validation dataset
    opt.isTest = False
    dataset_val = create_dataset(opt,data_folder)  # create a dataset_aligned given opt.dataset_mode and other options
    dataset_size_val = len(dataset_val)

    opt.isTrain = True #set Train back to true to enable training
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    evaluator_A = General_Evaluator(opt,dataset_size_train)
    evaluations = []
    random_indexes = []
    for _ in range(10):
        random_index = random.randint(0, dataset_size_train - 1)
        random_indexes.append(random_index)

    stop_training = False
    last_ten_losses = [100,100,100,100,100,100,100,100,100,100]

    original_num_threads = opt.num_threads
    original_batch_size = opt.batch_size
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        print(f'epoch: {epoch}')
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        opt.num_threads = original_num_threads
        opt.batch_size = original_batch_size
        train_loop(dataset_train,total_iters,opt,epoch_iter,model,visualizer,epoch,dataset_size_train,evaluator_A,evaluations,iter_data_time)
        stop_training,best_epoch = val_loop(model,dataset_val,opt,total_iters,visualizer,epoch,epoch_iter,dataset_size_val,iter_data_time,last_ten_losses,stop_training)

        #ADAPT SUCH THAT ONLY BEST MODEL IS SAVED :)
        if best_epoch:  # cache our model only if there is an improvement compared to last epoch
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        if stop_training:
            print('Used early stopping to end training')
            break
    print('Ran all the way to the maximal allowed number of epochs')