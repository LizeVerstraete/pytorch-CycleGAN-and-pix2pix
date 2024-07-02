import gc
import time
import matplotlib.pyplot as plt
import os

import torch


def train_loop(original_num_threads,original_batch_size,dataset,total_iters,opt,epoch_iter,model,visualizer,epoch,dataset_size,iter_data_time,evaluator,last_ten_losses,stop_training,save_dir,dataset_val,total_iters_val,epoch_iter_val,dataset_size_val,last_ten_evals_FID,last_ten_evals_WD,evaluator_test,evaluations_test,dataset_test):
    #print('GPU mem allocated begin train: ',torch.cuda.max_memory_allocated())
    stop_training = False
    model.train()
    opt.num_threads = original_num_threads
    opt.batch_size = original_batch_size
    opt.aligned = False
    for i, data in enumerate(dataset):  # inner loop within one epoch
        if i % int(20000/opt.batch_size) == 0:
            torch.cuda.empty_cache()
            gc.collect()
            #print('GPU mem allocated before val: ', torch.cuda.max_memory_allocated())
            stop_training,best_epoch = val_loop(evaluator,model,dataset_val,opt,total_iters_val,visualizer,epoch,epoch_iter_val,dataset_size_val,iter_data_time,last_ten_losses,stop_training,save_dir,last_ten_evals_FID,last_ten_evals_WD,total_iters,evaluator_test,evaluations_test,dataset_test)
            if stop_training:
                break
            torch.cuda.empty_cache()
            gc.collect()
            model.train()
            opt.num_threads = original_num_threads
            opt.batch_size = original_batch_size
            opt.aligned = False
            #print('GPU mem allocated after val: ', torch.cuda.max_memory_allocated())

        iter_start_time = time.time()  # timer for computation per iteration

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset_aligned and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            t_data = iter_start_time - iter_data_time
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id == 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
        #     losses = model.get_current_losses()
        #     current_loss = losses["cycle_A"]
        #     if current_loss <= best_loss:
        #         best_loss = current_loss
        #         print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        #         save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        #         model.save_networks(save_suffix)
        #     plt.show()

        iter_data_time = time.time()
    return total_iters, stop_training


def val_loop(evaluator,model,dataset,opt,total_iters,visualizer,epoch,epoch_iter,dataset_size,iter_data_time,last_ten_losses,stop_training,save_dir,last_ten_evals_FID,last_ten_evals_WD,total_iters_train,evaluator_test,evaluations_test,dataset_test):
    opt.aligned = False
    model.eval()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    sum_losses_epoch = 0
    number_losses = 0
    evaluator.reset(opt)

    for i, data in enumerate(dataset):
        if i>=(total_iters/20000*300)%dataset_size and i<(total_iters/20000*300+300)%dataset_size:
            iter_start_time = time.time()  # timer for computation per iteration
            #if i==0:
                #print('GPU mem allocated before set_input val: ', torch.cuda.max_memory_allocated())
            model.set_input(data)  # unpack data from data loader
            #if i==0:
                #print('GPU mem allocated after set_input val: ', torch.cuda.max_memory_allocated())
            #with torch.no_grad():
            #    model.forward()
            #    model.cycleA_loss()
            model.function_for_evaluation()
            #model.function_for_evaluation_no_backward()
            #with torch.no_grad():
            #    model.function_for_evaluation_no_backward()
            #if i==0:
                #print('GPU mem allocated after forward val: ', torch.cuda.max_memory_allocated())
            t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            #if total_iters % 1000 == 0:
            #     visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses,name="Test losses")
            sum_losses_epoch += float(getattr(model, 'loss_cycle_A'))
            number_losses += 1
            iter_data_time = time.time()
            #evaluator.process(model.fake_A.permute(0, 2, 3, 1), model.real_A.permute(0, 2, 3, 1), model.real_B.permute(0, 2, 3, 1),None)
    average_loss_epoch = sum_losses_epoch / number_losses
    for i in range(0,14):
        last_ten_losses[i] = last_ten_losses[i+1]
    last_ten_losses[14] = average_loss_epoch

    #if last_ten_losses[0] == min(last_ten_losses) and last_ten_losses.count(last_ten_losses[0]) == 1 and epoch > 1:
    #    stop_training = True
    print('validation losses: ', last_ten_losses)
    best_epoch = all(last_ten_losses[-1] <= value for value in last_ten_losses[:-1])
    if last_ten_losses[0] == min(last_ten_losses) and last_ten_losses.count(last_ten_losses[0]) == 1 and epoch > 1:
        stop_training = True

    #evaluation = evaluator.evaluate(opt)

    # for i in range(0,14):
    #     last_ten_evals_FID[i] = last_ten_evals_FID[i+1]
    #     last_ten_evals_WD[i] = last_ten_evals_WD[i+1]
    # last_ten_evals_FID[14] = evaluation["FID"]
    # last_ten_evals_WD[14] = evaluation["WD"]
    #
    # print('validation evaluations FID: ', last_ten_evals_FID)
    # print('validation evaluations WD: ', last_ten_evals_WD)

    adapted_save_dir = os.path.join(save_dir, f'epoch{epoch}_iter{total_iters_train}')
    # if not os.path.exists(adapted_save_dir):
    #     os.makedirs(adapted_save_dir)
    # with open(os.path.join(adapted_save_dir, f"evaluation.txt"), 'w') as f:
    #     f.write("Evaluation: " + str(evaluation))
    evaluator.reset(opt)

    if best_epoch:
        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters_train))
        save_suffix = 'epoch_%d_iter_%d' % (epoch, total_iters_train) if opt.save_by_iter else 'latest'
        model.save_networks(save_suffix)
        torch.cuda.empty_cache()
        gc.collect()
        #print('GPU mem allocated before test: ', torch.cuda.max_memory_allocated())
        test_loop(evaluator_test, evaluations_test, model, dataset_test, opt, save_dir,total_iters_train,epoch)
        torch.cuda.empty_cache()
        gc.collect()
        #print('GPU mem allocated after test: ', torch.cuda.max_memory_allocated())

    return stop_training,best_epoch

def test_loop(evaluator,evaluations,model,dataset,opt,save_dir,total_iters_train,epoch):
    opt.aligned = True
    model.eval()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    evaluator.reset(opt)
    index = 0
    adapted_save_dir = os.path.join(save_dir, f'epoch{epoch}_iter{total_iters_train}')
    save_dir_realX = os.path.join(adapted_save_dir, 'realX')
    save_dir_realY = os.path.join(adapted_save_dir, 'realY')
    save_dir_fakeX = os.path.join(adapted_save_dir, 'fakeX')
    save_dir_fakeY = os.path.join(adapted_save_dir, 'fakeY')
    if not os.path.exists(save_dir_fakeX):
        os.makedirs(save_dir_fakeX)
    if not os.path.exists(save_dir_realX):
        os.makedirs(save_dir_realX)
    if not os.path.exists(save_dir_fakeY):
        os.makedirs(save_dir_fakeY)
    if not os.path.exists(save_dir_realY):
        os.makedirs(save_dir_realY)

    print("Now saving tiles at: ", adapted_save_dir)
    for i, data in enumerate(dataset):
        index+=1
        model.set_input(data)
        with torch.no_grad():
             model.forward()
        #model.function_for_evaluation()

        realX_name = os.path.join(save_dir_realX, str(i))
        fakeX_name = os.path.join(save_dir_fakeX, str(i))
        realY_name = os.path.join(save_dir_realY, str(i))
        fakeY_name = os.path.join(save_dir_fakeY, str(i))

        realX = model.real_A[0].permute(1, 2, 0).cpu().detach().numpy()
        realX = (realX+1)/2
        plt.imsave(realX_name + ".png", realX)
        fakeX = model.fake_A[0].permute(1, 2, 0).cpu().detach().numpy()
        fakeX = (fakeX + 1) / 2
        plt.imsave(fakeX_name + ".png", fakeX)
        realY = model.real_B[0].permute(1, 2, 0).cpu().detach().numpy()
        realY = (realY + 1) / 2
        plt.imsave(realY_name + ".png", realY)
        fakeY = model.fake_B[0].permute(1, 2, 0).cpu().detach().numpy()
        fakeY = (fakeY + 1) / 2
        plt.imsave(fakeY_name + ".png", fakeY)

        evaluator.process(model.fake_A.permute(0, 2, 3, 1), model.real_A.permute(0, 2, 3, 1), model.real_B.permute(0, 2, 3, 1),None)
    evaluation = evaluator.evaluate(opt)
    print("test eval: ", evaluation)

    with open(os.path.join(adapted_save_dir, f"evaluation.txt"), 'w') as f:
        f.write("Evaluation: " + str(evaluation))

    evaluator.reset(opt)
    return