import time
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from src.data_processing.metrics_Boehringer_rewritten import unpaired_lab_WD_paths


def train_loop(dataset,total_iters,opt,epoch_iter,model,visualizer,epoch,dataset_size,evaluator_A,evaluations,iter_data_time,best_loss):
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

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
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id == 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            losses = model.get_current_losses()
            current_loss = losses["cycle_A"]
            if current_loss <= best_loss:
                best_loss = current_loss
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            plt.show()

        iter_data_time = time.time()

def val_loop(val_indexes,evaluator,evaluations,model,dataset,opt,total_iters,visualizer,epoch,epoch_iter,dataset_size,iter_data_time,last_ten_losses,stop_training):
    model.eval()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    sum_losses_epoch = 0
    number_losses = 0

    for i, data in enumerate(dataset):
        if i in val_indexes:
            iter_start_time = time.time()  # timer for computation per iteration
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            model.set_requires_grad([model.netD_A, model.netD_B], True)
            model.optimizer_D.zero_grad()
            model.backward_D_A()
            model.backward_D_B()
            model.set_requires_grad([model.netD_A, model.netD_B], False)  # Ds require no gradients when optimizing Gs
            model.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            model.backward_G()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size


            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id == 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses,name="Test losses")
                sum_losses_epoch += losses["cycle_A"]
                number_losses += 1
            iter_data_time = time.time()
            evaluator.reset(opt)
            evaluator.process(model.fake_A.permute(0, 2, 3, 1), model.real_A.permute(0, 2, 3, 1), None)
            evaluation = evaluator.evaluate()
            evaluations.append(evaluation)
            print("val eval: ", evaluation)
    print('val evals: ', evaluations)
    average_loss_epoch = sum_losses_epoch / number_losses
    for i in range(0,14):
        last_ten_losses[i] = last_ten_losses[i+1]
    last_ten_losses[14] = average_loss_epoch
    #x = np.arange(len(last_ten_losses))
    # Perform linear regression
    #slope, _, _, _, _ = linregress(x, last_ten_losses)
    #if slope >= 0:
    #    stop_training = True
    #print(slope)
    if last_ten_losses[0] == min(last_ten_losses) and last_ten_losses.count(last_ten_losses[0]) == 1:
        stop_training = True
    print(last_ten_losses)
    best_epoch = all(last_ten_losses[-1] <= value for value in last_ten_losses[:-1])
    return stop_training,best_epoch

def test_loop(evaluator,evaluations,model,dataset,opt,save_dir,epoch):
    model.eval()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    evaluator.reset(opt)
    index = 0

    for i, data in enumerate(dataset):
        index+=1
        model.set_input(data)
        model.function_for_evaluation()

        save_dir = os.path.join(save_dir, f'epoch{epoch}')
        save_dir_realX = os.path.join(save_dir, 'realX')
        save_dir_realY = os.path.join(save_dir, 'realY')
        save_dir_fakeX = os.path.join(save_dir, 'fakeX')
        save_dir_fakeY = os.path.join(save_dir, 'fakeY')
        if not os.path.exists(save_dir_fakeX):
            os.makedirs(save_dir_fakeX)
        if not os.path.exists(save_dir_realX):
            os.makedirs(save_dir_realX)
        if not os.path.exists(save_dir_fakeY):
            os.makedirs(save_dir_fakeY)
        if not os.path.exists(save_dir_realY):
            os.makedirs(save_dir_realY)

        realX_name = os.path.join(save_dir_realX, str(i))
        fakeX_name = os.path.join(save_dir_fakeX, str(i))
        realY_name = os.path.join(save_dir_realY, str(i))
        fakeY_name = os.path.join(save_dir_fakeY, str(i))

        print("Now saving tile with title: ", realX_name)
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

        evaluator.process(model.fake_A.permute(0, 2, 3, 1), model.real_A.permute(0, 2, 3, 1), None)
        evaluation = evaluator.evaluate()
        evaluations.append(evaluation)

        print("test eval: ", evaluation)
        #if index == 10:
        #    break
    sum_wb = 0
    sum_ssim = 0
    sum_psnr = 0
    for eval in evaluations:
        sum_wb += eval['WD']
        sum_ssim += eval['SSIM']
        sum_psnr += eval['PSNR']

    result = defaultdict(dict)
    avg_wb = unpaired_lab_WD_paths(save_dir_fakeY,save_dir_realY)

    avg_ssim = sum_ssim / len(evaluations)
    avg_psnr = sum_psnr / len(evaluations)
    result['avg_wdY'] = avg_wb
    result['avg_ssimY'] = avg_ssim
    result['avg_psnrY'] = avg_psnr

    print("Average WD:", avg_wb)
    print("Average SSIM:", avg_ssim)
    print("Average PSNR:", avg_psnr)

    with open(os.path.join(save_dir, f"final_evaluation_result_averaged.txt"), 'w') as f:
        f.write(str(result))

    print("test evals: ", evaluations)
    return