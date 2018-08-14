
import argparse
import io
import librosa
import numpy as np
import pandas as pd

def transcribe_file_with_word_time_offsets(speech_file):
    """Transcribe the given audio file synchronously and output the word time
    offsets."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='en-US',
        enable_word_time_offsets=True)

    response = client.recognize(config, audio)

    word_with_ts = []
    for result in response.results:
        #print result
        alternative = result.alternatives[0]
        print('Transcript: {}'.format(alternative.transcript))

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            word_with_ts.append((word ,start_time.seconds + start_time.nanos * 1e-9, end_time.seconds + end_time.nanos * 1e-9))
            #print('Word: {}, start_time: {}, end_time: {}'.format(
            #  word,
            #      start_time.seconds + start_time.nanos * 1e-9,
            #    end_time.seconds + end_time.nanos * 1e-9))
    return word_with_ts




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='File or GCS path for audio file to be recognized')
    args = parser.parse_args()
    filename = args.path
    data = pd.read_csv(filename, header = None)
    data = data.as_matrix()
    #print data

    google_speech_right_ans = 0
    google_speech_wrong_ans = 0
    for j in range(len(data)):
        print (data[j][0])
        word_with_ts = transcribe_file_with_word_time_offsets(data[j][0])
        max_rmse_value = 0
        max_rmse_word = "" 
        for i in word_with_ts:
            word = i[0]
            start_time = i[1]
            end_time = i[2]

            
            if start_time!=end_time:
                speech_vector, sr = librosa.load(data[j][0], offset = start_time, duration = (end_time - start_time))
                rmse =  librosa.feature.rmse(speech_vector)
                #print rmse.shape
                for k in rmse:
                    np_array_rmse = np.array(k)
                    max_value = np_array_rmse.max()
                    if max_value > max_rmse_value:
                        max_rmse_value = max_value
                        max_rmse_word = i[0]

        if max_rmse_word.upper() == data[j][1]:
            google_speech_right_ans+= 1
        else:
            google_speech_wrong_ans+=1
            print ("Predicted stress word: "+max_rmse_word.upper()+"\n"+"Actual stressed word : "+data[j][1] +"\n")
        
    print (str(float(google_speech_right_ans) / float(google_speech_right_ans + google_speech_wrong_ans) * 100))
