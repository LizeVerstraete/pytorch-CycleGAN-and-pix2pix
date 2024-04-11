from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from agents.helpers.Evaluator import General_Evaluator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    evaluator_A = General_Evaluator(dataset_size)
    evaluator_A.reset()
    for i, data in enumerate(dataset):
        model.set_input(data)
        #model.optimize_parameters()
        model.function_for_evaluation()
        #model.forward()
        print(i)
        evaluator_A.process(model.fake_A.permute(0, 2, 3, 1), model.real_A.permute(0, 2, 3, 1), None)
        if i % 20 == 0:
            plt.subplot(2, 2, 2)  # 1 row, 2 columns, subplot 1
            plt.imshow(model.real_B[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.title('Real B')

            plt.subplot(2, 2, 4)  # 1 row, 2 columns, subplot 2
            plt.imshow(model.fake_B[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.title('Fake B')

            plt.subplot(2, 2, 3)  # 1 row, 2 columns, subplot 2
            plt.imshow(model.fake_A[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.title('Fake A')

            plt.subplot(2, 2, 1)  # 1 row, 2 columns, subplot 2
            plt.imshow(model.real_A[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.title('Real A')

            plt.show()
        if i % 100 == 0:
            print(evaluator_A.evaluate())
    print(evaluator_A.evaluate())