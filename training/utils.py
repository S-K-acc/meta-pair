import itertools
import numpy as np
import tensorflow as tf
import keras


def get_matchdataset(match_data):

    match_data = match_data.T  

    phi = match_data[:, 0:150]        
    cphi = match_data[:, 150:300]      
    rho = match_data[:, 300:701]       

    training_inputs_c1b = {
        "rho": rho.astype(np.float32),
        "phi": phi.astype(np.float32)
    }

    training_output_c1b = {
        "cphi": cphi.astype(np.float32)
    }

    dataset = tf.data.Dataset.from_tensor_slices((training_inputs_c1b, training_output_c1b))
    return dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True).repeat(32).batch(64).prefetch(tf.data.AUTOTUNE)
    
    
class DataGenerator(keras.utils.PyDataset):
    def __init__(self, meta_data, batch_size=256, shuffle=True):
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(meta_data.T))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / (self.batch_size)))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.meta_data.T[batch_indices]

        rho = batch_data[:, 0:401].astype(np.float32)
        phi = batch_data[:, 401:551].astype(np.float32)
        c1 = batch_data[:, 551:552].astype(np.float32)

        inputs = {"rho": rho, "phi": phi}
        outputs = {"c1": c1}
        return inputs, outputs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
            
def get_dataset_c1(trainingGenerator):	
    def gen():
        for i in range(len(trainingGenerator)):
            yield trainingGenerator[i]

    return tf.data.Dataset.from_generator(gen, output_signature=(
        {
            "rho": tf.TensorSpec(shape=(trainingGenerator.batch_size, 401), dtype=tf.float32),
            "phi": tf.TensorSpec(shape=(trainingGenerator.batch_size, 150), dtype=tf.float32),
        },
        {
            "c1": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
        }
    )).repeat(2).prefetch(tf.data.AUTOTUNE)

