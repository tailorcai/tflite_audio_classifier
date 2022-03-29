import numpy as np
import tensorflow as tf
import os, csv, sys
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# from tflite_support import metadata as _metadata
# from tflite_support import metadata_schema_py_generated as _metadata_fb
from IPython import display

class TFLiteModel:
    def __init__(self, model_path):
        super(TFLiteModel, self).__init__()
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        # print( self.input_details )
        self.output_details = self.interpreter.get_output_details()
        # print( self.output_details )
        # [{'name': 'input', 'index': 0, 'shape': array([  8, 224, 224,   3], dtype=int32), 'dtype': <class 'numpy.uint8'>, 'quantization': (1.0, 0)}]
        # [{'name': 'output', 'index': 172, 'shape': array([   8, 1001], dtype=int32), 'dtype': <class 'numpy.uint8'>, 'quantization': (0.09889253973960876, 58)}]
        self.data_type = self.input_details[0]["dtype"]
        self.data_size = (15600,)
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

    def __call__(self, waveform):
        return self.predict(waveform)

    def predict(self, waveform):
        self.interpreter.set_tensor(self.input_index, waveform)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_index).flatten()
        # print( output_data )
        preds = output_data.argsort()[::-1][:5]
        return preds


def evaluate(models, names, class_map_path):
    for n in names:
        test_file = tf.io.read_file(n)
        test_audio, _ = tf.audio.decode_wav(contents=test_file)
        # print( test_audio.shape)
        waveform = tf.squeeze(test_audio, axis=-1)
        # print(waveform[:100])
        preds = models[0]( waveform[10000:15600+10000])
        print( n )
        print( [class_map_path[p] for p in preds] )
    # display.display(display.Audio(waveform, rate=16000))

    # spectrogram = get_spectrogram(waveform)
    # print( spectrogram.shape )

    # fig, axes = plt.subplots(2, figsize=(12, 8))
    # timescale = np.arange(waveform.shape[0])
    # axes[0].plot(timescale, waveform.numpy())
    # axes[0].set_title('Waveform')
    # axes[0].set_xlim([0, 16000])

    # plot_spectrogram(spectrogram.numpy(), axes[1])
    # axes[1].set_title('Spectrogram')
    # plt.show()

def plot_spectrogram(spectrogram, ax):
    # if len(spectrogram.shape) > 2:
    # assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def load_class_map(fn):
    with open(fn, newline='') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        return [ row['display_name'] for row in spamreader]
            
if __name__ == '__main__':
    # model_path_0 = "mobilenet_v2_1.0_224_frozen.pb"
    # model_0 = TFModel(model_path_0)
    #
    # model_path_1 = "mobilenet_v2_1.0_224.tflite"
    # model_1 = TFLiteModel(model_path_1)
    #
    # model_path_2 = "mobilenet_v2_1.0_224_quant_frozen_opt.pb"
    # model_2 = TFModel(model_path_2)
    #
    # model_path_3 = "mobilenet_v2_1.0_224_quant.tflite"
    # model_3 = TFLiteModel(model_path_3)
    #
    # models = [model_0, model_1, model_2, model_3]
    # names = [model_path_0, model_path_1, model_path_2, model_path_3]
    # evaluate(models, names)

    model_path = "lite-model_yamnet_classification_tflite_1.tflite"
    model = TFLiteModel(model_path)

    class_map_path = load_class_map('yamnet_class_map.csv')
    # class_map_path = model.class_map_path().numpy()
    # print(class_map_path)
    evaluate([model], sys.argv[1:], class_map_path)
