import time
import random
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from agents.helpers.Evaluator_nonAligned import General_Evaluator
from agents.helpers.Evaluator_aligned import General_Evaluator as General_Evaluator_aligned
from results.traintestval_functions import train_loop, val_loop, test_loop
import os

if __name__ == '__main__':
    #SET VARIABLES
    data_folder = r"/esat/biomeddata/kkontras/r0786880/biopsy_data_bigger_dataset_412_entropy_norm11"
    data_folder_test = r"/esat/biomeddata/kkontras/r0786880/biopsy_data_manually_aligned_412"
    print_intermediate_results = False
    save_dir = r"/esat/biomeddata/kkontras/r0786880/results/evaluation_some_training_small_tiles"
    #DON'T EDIT ANYTHING BELOW
    opt,message = TrainOptions().parse()   # get training options
    original_save_dir = os.path.join(save_dir,opt.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    #opt.aligned = True
    dataset_train = create_dataset(opt,data_folder)  # create a dataset_aligned given opt.dataset_mode and other options
    dataset_size_train = len(dataset_train)    # get the number of images in the dataset_aligned.
    print('The number of training images = %d' % dataset_size_train)
    opt.isTrain = False #set Train and Test to false to create validation dataset
    opt.isTest = False
    dataset_val = create_dataset(opt,data_folder)  # create a dataset_aligned given opt.dataset_mode and other options
    dataset_size_val = len(dataset_val)
    print('The number of validation images = %d' % dataset_size_val)

    opt.allData = True
    opt.aligned = True
    dataset_test = create_dataset(opt,data_folder_test)  # create a dataset_aligned given opt.dataset_mode and other options
    dataset_size_test = len(dataset_test)
    print('The number of test images = %d' % dataset_size_test)

    opt.isTrain = True #set Train back to true to enable training
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    total_iters_val = 0
    evaluator_train = General_Evaluator(opt,dataset_size_train)
    evaluations_train = []
    evaluator_val = General_Evaluator(opt,dataset_size_val)
    evaluations_val = []
    evaluator_test = General_Evaluator_aligned(opt,dataset_size_test)
    evaluations_test = []
    random.seed(42)

    stop_training = False
    last_fifteen_losses = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    last_fifteen_evals_FID = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    last_fifteen_evals_WD = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
    best_loss = 100

    original_num_threads = opt.num_threads
    original_batch_size = opt.batch_size
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        print(f'epoch: {epoch}')
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter_val = 0
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        opt.num_threads = original_num_threads
        opt.batch_size = original_batch_size
        opt.aligned = False
        total_iters, stop_training = train_loop(original_num_threads,original_batch_size,dataset_train,total_iters,opt,epoch_iter,model,visualizer,epoch,dataset_size_train,iter_data_time,evaluator_val,last_fifteen_losses,stop_training, original_save_dir,dataset_val,total_iters_val,epoch_iter_val,dataset_size_val,last_fifteen_evals_FID,last_fifteen_evals_WD,evaluator_test,evaluations_test,dataset_test)
        #stop_training,best_epoch = val_loop(evaluator_val,model,dataset_val,opt,total_iters_val,visualizer,epoch,epoch_iter_val,dataset_size_val,iter_data_time,last_fifteen_losses,stop_training, original_save_dir,last_fifteen_evals_FID,last_fifteen_evals_WD,total_iters,evaluator_test,evaluations_test,dataset_test)
        opt.aligned = True
        #test_loop(evaluator_test,evaluations_test,model,dataset_test,opt,original_save_dir,total_iters,epoch)

        if not opt.use_scheduler:
            model.update_learning_rate()    # update learning rates at the end of every epoch.

        #ADAPT SUCH THAT ONLY BEST MODEL IS SAVED :)
        # if best_epoch:  # cache our model only if there is an improvement compared to last epoch
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        if stop_training:
            print('Used early stopping to end training')
            break
    if not stop_training:
        print('Ran all the way to the maximal allowed number of epochs')