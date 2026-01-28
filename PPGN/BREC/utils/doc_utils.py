import json
import os
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.abspath("../experiments")


def write_to_file_doc(train_acc, train_loss, val_acc, val_loss, epoch, config):
    val = [config.exp_name, epoch, train_loss, train_acc, val_loss, val_acc, config.timestamp]
    columns = ['experiment_name', 'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'timestamp']
    
    fullpath = os.path.join(config.summary_dir, 'per_epoch_stats.csv')
    if os.path.exists(fullpath):
        f = pd.read_csv(fullpath)
    else:
        f = pd.DataFrame(columns=columns)
    new_row = pd.DataFrame([val], columns=columns)
    f = pd.concat([f, new_row], ignore_index=True)
    f.to_csv(fullpath, index=False)



def doc_used_config(config):
    """
    Log the used config file in the summary directory in JSON format
    :param config: configuration EasyDict
    :return:
    """
    fullpath = os.path.join(config.summary_dir, 'used_config.json')

    with open(fullpath, 'w') as fp:
        json.dump(config, fp)


def create_experiment_results_plot(title, parameter, directory, log=False):
    fullpath = os.path.join(directory, 'per_epoch_stats.csv')
    
    df = pd.read_csv(filepath_or_buffer=fullpath)
    epochs = df["epoch"]
    train_param = df["train_{0}".format(parameter)]
    val_param = df["val_{0}".format(parameter)]
    
    if log:
        plt.semilogy(epochs, train_param, 'r', label='train')
        plt.semilogy(epochs, val_param, 'b', label='validation')
        plt.ylabel(parameter + " semilog")
        axes = plt.gca()
        axes.set_ylim([10 ** -3, 10 ** 1])
    else:
        plt.plot(epochs, train_param, 'r', label='train')
        plt.plot(epochs, val_param, 'b', label='validation')
        plt.ylabel(parameter)
        axes = plt.gca()
        axes.set_ylim([0.1, 0.95])
    plt.xlabel('Epochs')

    plt.title(title)
    plt.legend()
    file_name = (os.path.join(directory, (title + parameter + ".png")))
    plt.savefig(file_name)

    plt.close()
    return file_name

def summary_10fold_results(summary_dir):
    df = pd.read_csv(os.path.join(summary_dir, 'per_epoch_stats.csv'))
    df = df.drop(['train_loss', 'train_accuracy', 'timestamp', 'experiment_name'], axis=1)
    df_group_std = df.groupby('epoch').std()
    df_group = df.groupby('epoch').mean()
    df_group['std'] = df_group_std.val_accuracy
    best_epoch = df_group['val_accuracy'].idxmax()
    
    best_row = df_group.loc[best_epoch]
    print("Results")
    print("Best epoch - {0}".format(best_epoch))
    print("Mean Accuracy = {0}".format(best_row['val_accuracy']))
    print("Mean std = {0}".format(best_row['std']))
    
    
    df2 = df[df.epoch == best_epoch].copy()
    df2['fold'] = pd.Series(range(10), index=df2.index) + 1
    df2 = df2.append(pd.Series([best_epoch, best_row['val_loss'], best_row['val_accuracy'], 'mean'], index=df2.columns),
                     ignore_index=True)
    fullpath = os.path.join(summary_dir, 'exp_summary.csv')
    df2.to_csv(fullpath, index=False)
    
    
def summary_qm9_results(summary_dir, test_dists, test_loss, best_epoch, create_csv=True):
    
    print("Results")
    print("\tTest absolute distances by best epoch ({0}) of validation = \n{1}".format(best_epoch, test_dists))
    
    if not create_csv:
        return

    # Write two lines to csv file
    test_dists = list(test_dists)
    if len(test_dists) > 1:
        # 12 targets
        columns = ['test_loss'] + ['dist{}'.format(i) for i in range(1, 13)]
    else:
        columns = ['test_loss', 'dist']
    values1 = [test_loss] + test_dists

    fullpath = os.path.join(summary_dir, 'exp_summary.csv')
    pd.DataFrame([values1], columns=columns).to_csv(fullpath, index=False)
    
    