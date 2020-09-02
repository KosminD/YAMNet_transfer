import sys
sys.path.append("../")
import tensorflow as tf

from tensorflow.keras.models import load_model
from yamnet import yamnet_frames_model_transfer, yamnet_frames_model
from params import Params
import tensorflow as tf
import os
import time
import librosa
import numpy as np

params = Params()
print(params.tflite_compatible)
def convert_general(general_model):
    #general = load_model(general_model)
    model_general = yamnet_frames_model(params)
    model_general.load_weights('/home/pc/PycharmProjects/yamnet_medium/output/yamnet.tflite', by_name=True)
    print(model_general.summary())

    converter = tf.lite.TFLiteConverter.from_keras_model(model_general)
    tflite_model = converter.convert()
    open("general_model.tflite", "wb").write(tflite_model)

def read_wav(fname, output_sr, use_rosa=True):

    waveform, sr = librosa.load(fname, sr=output_sr)

    return waveform.astype(np.float64)

def main():
    interpreter = tf.lite.Interpreter(model_path="/home/pc/PycharmProjects/yamnet_medium/output/yamnet.tflite")
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    print(inputs)
    outputs = interpreter.get_output_details()
    st = time.time()
    for file in os.listdir(
            "/home/pc/Downloads/training-pipeline-stage(1)/training-pipeline-stage/data/CatSounds/come_in"):
        waveform = read_wav(
            "/home/pc/Downloads/training-pipeline-stage(1)/training-pipeline-stage/data/CatSounds/come_in/" + file,
            16000)[:15600]

        # Predict YAMNet classes.
        waveform = np.array(waveform, dtype=np.float32)
        print(waveform.shape)
        audio_input_index = interpreter.get_input_details()[0]['index']
        scores_output_index = interpreter.get_output_details()[0]['index']
        embeddings_output_index = interpreter.get_output_details()[1]['index']

        interpreter.resize_tensor_input(audio_input_index, [len(waveform)], strict=True)
        interpreter.allocate_tensors()
        interpreter.set_tensor(audio_input_index, waveform)
        interpreter.invoke()
        print(interpreter.get_tensor(embeddings_output_index))
        scores = interpreter.get_tensor(outputs[0]['index'])
    print(time.time()-st)
#convert_general("models/test.h5")
main()
