from time import time
import os
from glob import glob
import pickle
from DataGenerator import DataGenerator
import tensorflow as tf


def main():
    start_time = time()
    base_dir = '/home/ubuntu/uk-biobank-ecg_mount/'
    wave_save_dir = os.path.join(base_dir, '220408_uk-biobank-ecg_waveform')

    # X
    csv_files = glob(os.path.join(wave_save_dir, '*.csv'))
    # Y
    with open('UKBiobank_labels.pickle', 'rb') as f:
        labels = pickle.load(f)
    # load trained model
    model = tf.keras.models.load_model("model-best.h5")

    # DataGenerator
    test_generator = DataGenerator(csv_files, labels, 64) #shuffle=False
    # model predict
    test_predictions = model.predict(test_generator)
    loss, mae, mse = model.evaluate(test_generator, verbose=2)

    with open('best-model_ukbiobank_pred.pickle', 'wb') as f:
        pickle.dump(test_predictions, f)

    print("---{}s seconds---".format(time()-start_time))



if __name__ == "__main__":
	main()
