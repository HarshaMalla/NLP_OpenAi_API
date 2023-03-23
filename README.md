# About this Repo 💻

This repository showcases the incredible power of OpenAI's API for Natural Language Processing. With the help of OpenAI's Whisper API, I was able to perform an impressive task - converting various forms of audio, such as speech, conversation, and even music, into text format. But that's not all - I then utilized OpenAI's powerful Davinci language model to transform those text files into AI-generated text of unparalleled quality. And to bring those words to life, I employed the Google Text to Speech library to convert the text back into audio, producing stunningly realistic results.



## Objectives

The objective of this project is to explore the capabilities of the Whisper API and OpenAI's Davinci language model for Natural Language Processing (NLP) by building an application that can accurately convert various types of audio inputs (speeches, songs, and conversations) into text format. The goal is to measure the accuracy of the transcription against the original audio input and then utilize the OpenAI Davinci model to generate a logical response based on the text output. Finally, the project aims to convert the text output back into audio format using text-to-speech (TTS) software or libraries.



## Tech Stack
![jhghd](https://user-images.githubusercontent.com/96933166/227278411-7b3256cd-f3cb-4860-a729-ed2ea14a9c7d.png)

 



<img title="" src="" alt="jhghd.png" width="728">

**Open AI's Whisper API :** For converting various types of audio inputs (speeches, songs, and conversations) into text format.

**OpenAI's Davinci language model :** For generating AI-generated text based on the transcribed text output.

**Python :** Used the Python programming language for programming of the application

**Jupyter :** used Jupyter to experiment and test the application

 

## Methodology :

### Data Collection and Pre-processing

- Audio data collection: Collected Speech data from Mozilla Audio Datasets (without noise ), 10-sec youtube clips which are converted to mp3 format (with noise), Music from Youtube Music and Conversation data from Youtube Podcasts and Internet Archive 
- Audio Data Pre-Processing: 
  
  --> Speech Data : Converted Youtube mp4 clips to mp3 using 4k video converter.
  
  --> Music : Converted Youtube mp4 clips to mp3 using 4k video converter.
  
  --> Conversation: Converted Youtube mp4 clips to mp3 using a4k video converter and trimmed the long podcast clips into shorter chunks using Audacit .
  
  ### Data cleaning and formatting
  
  Manually deleted some empty audio files and some corrupted files and made sure that every audio file is in either ".MP3" format or ".WAV" format.
  
  ### Audio-to-Text Conversion
  
  **Whisper API :** The Whisper API is an API developed by OpenAI that allows developers to easily convert spoken audio into text format. The API leverages machine learning algorithms to accurately transcribe audio inputs, including speeches, songs, and conversations. The Whisper API is designed to handle real-time audio inputs, which means it can transcribe audio in near real-time, making it ideal for applications that require live transcription
  
  
  
  *Importing Whisper APi (Offline)*
  
  ```py
  
  #Using Offline Module 
  import whisper
  
  model = whisper.load_model("base")
  
  # load audio and pad/trim it to fit 30 seconds
  audio = whisper.load_audio("audio.mp3")
  audio = whisper.pad_or_trim(audio)
  
  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(audio).to(model.device)
  
  # detect the spoken language
  _, probs = model.detect_language(mel)
  print(f"Detected language: {max(probs, key=probs.get)}")
  
  # decode the audio
  options = whisper.DecodingOptions(fp16 = False)
  result = whisper.decode(model, mel, options)
  
  # print the recognized text
  print(result.text)
  
  
  ```
  
  *But for our projec,t we used Open Ai's Whisper API :*
  
  ```py
  import openai
  
  openai.api_key = 'Your_API_Key'
  model_id = 'whisper-1'
  
  media_file_path = ''
  media_file = open(media_file_path, 'rb')
  
  response = openai.Audio.transcribe(
      api_key=openai.api_key,
      model=model_id,
      file=media_file
  )
  # Add 'data' attribute to response object
  response.data = {'text': response.text}
  
  print(response.data['text'])
  ```
  
  

*Created separate directories for separate type of Audio Data sets.*

![](C:\Users\91944\AppData\Roaming\marktext\images\2023-03-23-21-14-21-image.png)

*After creating seperate directories, The files in the directories are converted into text files using Whisper APi, and saved the output files in separate folders based on their audio type.*

```py
#Converting MP3 into .Text format and store them in different directories

import os
import openai

openai.api_key = 'Your_API_Key'
model_id = 'whisper-1'

# Create output directories
os.makedirs('Music_txt', exist_ok=True)
os.makedirs('Speech_txt', exist_ok=True)
os.makedirs('Conversation_txt', exist_ok=True)

# Loop over input directories
for dir_name in ['Music', 'Speech', 'Conversation']:
    # Loop over audio files in directory
    for filename in os.listdir(f'Audio_dataset/{dir_name}'):
        # Skip any non-audio files
        if not filename.endswith('.mp3'):
            continue

        # Transcribe audio file
        with open(f'Audio_dataset/{dir_name}/{filename}', 'rb') as f:
            response = openai.Audio.transcribe(
                api_key=openai.api_key,
                model=model_id,
                file=f
            )

        # Save transcription to output file
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = f'{dir_name}_txt/{output_filename}'
        with open(output_path, 'w') as f:
            f.write(response['text'])
 
```

### 

### Measuring transcription accuracy :

To assess the accuracy of the transcription process, the Levenshtein algorithm is typically used to compare the original transcription with the output generated by the Whisper API. However, in this particular project, due to unthe availability of the original transcription data, it is not possible to measure the accuracy of the transcription process against the original data.



Based on my analysis and experience, the Whisper model exhibits higher accuracy levels when transcribing audio files that have minimal background noise and distortion, as compared to audio files with significant levels of noise. The algorithm is designed to identify and isolate voice signals, and any background noise or interference may cause inaccuracies in the transcription output. However, it is worth noting that the Whisper API is still a state-of-the-art technology and can effectively handle a wide range of audio inputs with a high level of accuracy



### Text-to-AI Generated Text Conversion

**OpenAI Davinci Model:** The OpenAI Davinci Model is a state-of-the-art language model developed by OpenAI, which leverages deep neural networks to generate high-quality natural language responses. The model is designed to understand the semantic and syntactic structures of human language and can generate coherent and contextually appropriate responses.

**Natural Language Processing (NLP) techniques:** In the context of the OpenAI Davinci model, NLP techniques are used to analyze and understand the content of the text input, and generate logical responses that are coherent and contextually appropriate. This involves a deep understanding of the semantics and syntax of natural language, and the ability to generate text that is grammatically correct, contextually appropriate, and stylistically consistent.



*The code imports the OpenAI library and sets the API key to authorize API calls. The Davinci model is selected as the model engine. The input text is read from a text file and sent to the Davinci model for processing using the OpenAI Completion API.*

```py
import openai
import os

openai.api_key = 'Your_API_KEY'
model_engine = 'davinci'
input_file_path = 'Converted_txt\Conversation_txt\Learn English 15min podcast.txt'

# Read input text from file
with open(input_file_path, 'r') as f:
    input_text = f.read().strip()

# Send input text to OpenAI Davinci model for processing
response = openai.Completion.create(
    engine=model_engine,
    prompt=input_text,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=1,
)

# Print the generated response
print(response.choices[0].text.strip())
```

### 

### Generating logical responses

The process of generating logical responses typically involves several steps, such as pre-processing the input text, analyzing and understanding the content of the input text, generating a logical response, and post-processing the response to ensure that it is grammatically correct and contextually appropriate. The OpenAI Davinci model performs all of these steps automatically, allowing users to generate high-quality natural language responses with minimal effort.



*This code uses OpenAI's Davinci model to generate AI-generated text based on input text files. The generated text is then saved in a separate directory for each type of input text file (music, speech, or conversation).*



```py
import openai
import os

openai.api_key = ''

# Define the prompt to input to the OpenAI Davinci model
prompt = "\n\n"

# Create the response texts directories if they do not exist
response_dirs = {'Music_txt': 'music_responses', 'Speech_txt': 'speech_responses', 'Conversation_txt': 'conversation_responses'}
for response_dir in response_dirs.values():
    if not os.path.exists(response_dir):
        os.mkdir(response_dir)

# Loop over the transcribed input files
for dir_name, response_dir in response_dirs.items():
    for filename in os.listdir(dir_name):
        if not filename.endswith('.txt'):
            continue

        # Read the input text file
        with open(os.path.join(dir_name, filename), 'r', encoding='utf-8') as f:
            input_text = f.read().strip()

        # Add the input text to the prompt
        prompt += f"{input_text}\n\n"

        # Send the prompt to the OpenAI Davinci model for processing
        response = openai.Completion.create(
            engine='davinci',
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=1,
        )

        # Save the generated response to a separate file in response_texts directory
        output_filename = os.path.splitext(filename)[0] + '_response.txt'
        output_path = os.path.join(response_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.choices[0].text.strip())

        # Reset the prompt to its original value
        prompt = "\n\n"
```

### Text-to-Audio Conversion

Text-to-Audio Conversion is the process of converting written text into spoken words. This process involves the use of Text-to-Speech (TTS) technology, which can generate synthetic speech from written text.



**Text-to-speech (TTS) software :** I initially used the Google Text-to-Speech (TTS) library to convert text into audio and saved the audio files in separate directories. However, I ran into an error due to excessive use of the Google TTS API. To mitigate this issue, I implemented a backup code that uses the 'pyttsx3' library to convert text into audio in case the Google TTS API fails



*Converting text files to audio using Google Text-to-Speech Library and saving in "Text_to_speech" directory.*

```py
from gtts import gTTS
import os

# Set the languages for the text-to-speech conversion
languages = {
    'music_responses': 'en',
    'speech_responses': 'en',
    'conversation_responses': 'en'
}

# Set the output directories for the audio files
output_dirs = {
    'music_responses': 'Text_to_speech/music',
    'speech_responses': 'Text_to_speech/speech',
    'conversation_responses': 'Text_to_speech/conversation'
}

# Create the output directories if they don't exist
for output_dir in output_dirs.values():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Loop over the text files in the directories
for dir_name, lang in languages.items():
    input_dir = os.path.join('Responses_txt', dir_name)
    output_dir = output_dirs[dir_name]
        
    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue

        # Read the contents of the file
        with open(os.path.join(input_dir, filename), 'r', encoding="utf-8") as f:
            contents = f.read().strip()
            
        # Check if contents are not empty
        if contents:
            # Convert the text to speech
            tts = gTTS(text=contents, lang=lang)
            audio_file = os.path.splitext(filename)[0] + '.mp3'
            audio_path = os.path.join(output_dir, audio_file)

            # Save the audio file
            tts.save(audio_path)
```

*Code for Text-to-Speech Conversion with Backup using Google Text-to-Speech and pyttsx3 Libraries*

```py
import time
import os
from gtts import gTTS
import pyttsx3

# Set the languages for the text-to-speech conversion
languages = {
    'music_responses': 'en',
    'speech_responses': 'en',
    'conversation_responses': 'en'
}

# Set the output directories for the audio files
output_dirs = {
    'music_responses': 'Text_to_speech/music',
    'speech_responses': 'Text_to_speech/speech',
    'conversation_responses': 'Text_to_speech/conversation'
}

# Create the output directories if they don't exist
for output_dir in output_dirs.values():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Loop over the text files in the directories
for dir_name, lang in languages.items():
    input_dir = os.path.join('Responses_txt', dir_name)
    output_dir = output_dirs[dir_name]
        
    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue

        # Check if the file has already been converted
        audio_file = os.path.splitext(filename)[0] + '.mp3'
        audio_path = os.path.join(output_dir, audio_file)
        if os.path.exists(audio_path):
            continue
        
        # Read the contents of the file
        with open(os.path.join(input_dir, filename), 'r', encoding="utf-8") as f:
            contents = f.read().strip()
            
        # Check if contents are not empty
        if contents:
            # Try to convert the text to speech with gtts
            try:
                tts = gTTS(text=contents, lang=lang)
                tts.save(audio_path)
            except Exception as e:
                print(f"Error converting {filename} with gtts: {str(e)}")
                print(f"Falling back to pyttsx3...")
                # Try to convert the text to speech with pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.7)
                engine.save_to_file(contents, audio_path)
                engine.runAndWait()
```

### Evaluation metrics

1. Word Error Rate (WER): This measures the percentage of words that are incorrectly recognized by the speech recognition system. WER is calculated by dividing the number of incorrect words by the total number of words in the transcription.

2. Character Error Rate (CER): This measures the percentage of characters that are incorrectly recognized by the speech recognition system. CER is calculated by dividing the number of incorrect characters by the total number of characters in the transcription.

3. Levenshtein Distance (LD): This measures the number of insertions, deletions, and substitutions required to transform the recognized text into the ground truth text. LD is calculated as the minimum number of edit operations required to transform one string into the other.

4. Accuracy: This measures the percentage of correctly recognized words or characters in the transcription. Accuracy is calculated by dividing the number of correct words or characters by the total number of words or characters in the transcription.

### Conclusion

In conclusion, the project aimed to transcribe audio files using the Whisper API and store the transcriptions in text files. The transcriptions were then converted to audio files using Google's text-to-speech API and saved in separate directories. However, due to the absence of original transcriptions of audio files, accuracy metrics could not be calculated to evaluate the performance of the Whisper API. In case the Whisper API fails, the project also includes a backup code that uses the pyttsx3 library for text-to-speech conversion. Overall, the project demonstrates how to use API services and libraries to automate audio transcription and text-to-speech conversion tasks.
