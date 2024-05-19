import os
import glob
import numpy as np
import yaml
import logging
from tqdm import tqdm
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense
from utils import *
from visualize import visualizer

"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def keras_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64)
    """
    inputLayer = Input(shape = (inputDim ,))
    h = Dense(64, activation = "relu")(inputLayer)
    h = Dense(64, activation = "relu")(h)
    h = Dense(8, activation = "relu")(h)
    h = Dense(64, activation = "relu")(h)
    h = Dense(64, activation = "relu")(h)
    h = Dense(inputDim, activation = None)(h)

    return Model(inputs = inputLayer, outputs = h)

if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok = True)
    os.makedirs(param["model_directory"], exist_ok = True)
    os.makedirs(param["result_directory"], exist_ok = True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath(f"{param["base_directory"]}")))

    # setup the result
    result_file = f"{param["result_directory"]}/{param[["result_file"]]}"
    results = {}

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print(f"[{dir_idx + 1}/{len(dirs)}] {target_dir}")

        # dataset param        
        db = param["dataset"]["db_name"]
        machine_rpm = param["dataset"]["machine_rpm"]
        machine_id = param["dataset"]["machine_id"]

        # setup path
        evaluation_result = {}
        
        train_pickle = f"{param["pickle_directory"]}/train_{machine_rpm}_{machine_id}_{db}.pickle"
        eval_files_pickle = f"{param["pickle_directory"]}/eval_files_{machine_rpm}_{machine_id}_{db}.pickle"
        eval_labels_pickle = f"{param["pickle_directory"]}/eval_labels_{machine_rpm}_{machine_id}_{db}.pickle"

        model_file = f"{param["model_directory"]}/model_{machine_rpm}_{machine_id}_{db}.hdf5"

        history_img = f"{param["model_directory"]}/history_{machine_rpm}_{machine_id}_{db}.png"

        roc_img = f"{param["model_directory"]}/ROC_{machine_rpm}_{machine_id}_{db}.png"

        evaluation_result_key = f"{machine_rpm}_{machine_id}_{db}"

        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        # If pickled files exist, use them or generate them
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_data = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir,
                                                                                   abnormal_samples = param["dataset"]["abnormal_samples"],
                                                                                   normal_dir_name = param["dataset"]["normal_dir"],
                                                                                   abnormal_dir_name = param["dataset"]["abnormal_dir"])

            train_data = list_to_vector_array(train_files,
                                              msg = "generate train_dataset",
                                              n_mels = param["feature"]["n_mels"],
                                              frames = param["feature"]["frames"],
                                              n_fft = param["feature"]["n_fft"],
                                              hop_length = param["feature"]["hop_length"],
                                              power = param["feature"]["power"])

            save_pickle(train_pickle, train_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        # model training
        print("============== MODEL TRAINING ==============")
        model = keras_model(param["feature"]["n_mels"] * param["feature"]["frames"])
        model.summary()

        # training
        if os.path.exists(model_file):
            model.load_weights(model_file)
        else:
            model.compile(**param["fit"]["compile"])
            history = model.fit(train_data,
                                train_data,
                                epochs = param["fit"]["epochs"],
                                batch_size = param["fit"]["batch_size"],
                                shuffle = param["fit"]["shuffle"],
                                validation_split = param["fit"]["validation_split"],
                                verbose = param["fit"]["verbose"])

            visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
            visualizer.save_figure(history_img)
            model.save_weights(model_file)

        # evaluation
        print("============== EVALUATION ==============")
        y_score = [0. for k in eval_labels]
        y_true = eval_labels

        for num, file_name in tqdm(enumerate(eval_files), total = len(eval_files)):
            try:
                data = file_to_vector_array(file_name,
                                            n_mels = param["feature"]["n_mels"],
                                            frames = param["feature"]["frames"],
                                            n_fft = param["feature"]["n_fft"],
                                            hop_length = param["feature"]["hop_length"],
                                            power = param["feature"]["power"])
                
                error = np.mean(np.square(data - model.predict(data)), axis = 1)
                
                y_score[num] = np.mean(error)
            except:
                logger.warning(f"File broken!!: {file_name}")

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        visualizer.roc_plot(fpr, tpr) 
        visualizer.save_figure(roc_img)

        score = metrics.roc_auc_score(y_true, y_score)
        logger.info(f"AUC : {score}")
        evaluation_result["AUC"] = float(score)

        results[evaluation_result_key] = evaluation_result
        print("===========================")

    # output results
    print("\n===========================")
    logger.info(f"all results -> {result_file}")
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")