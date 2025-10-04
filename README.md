# Speech Emotion Recognition

This project detects human emotions (happy, sad, angry, neutral) from speech audio using machine learning and deep learning techniques.

## Dataset
The dataset used in this project can be downloaded from Kaggle:  
[Speech Emotion Recognition Dataset]
(https://www.kaggle.com/code/aishwaryaljadhav/speech-emotion-recognition)

After downloading, place the files in a folder called `data/` inside the project:

## Tech Stack
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Librosa, TensorFlow, Keras  
- **Tools:** Jupyter Notebook  
- **Model:** CNN / LSTM for emotion classification  
- **Features Used:** MFCC (Mel Frequency Cepstral Coefficients)

## Steps Performed
1. **Data Collection:** Downloaded the speech emotion dataset from Kaggle.  
2. **Data Preprocessing:**  
   - Extracted MFCC features from audio files.  
   - Split data into train, test, and validation sets.  
3. **Model Building:**  
   - Designed a CNN/LSTM model for emotion recognition.  
   - Trained the model on the preprocessed dataset.  
4. **Evaluation:**  
   - Calculated accuracy and visualized the confusion matrix.  
   - Tested predictions on new audio samples.

## Results
- Achieved an accuracy of **~85%** on the test set.  
- Confusion matrix shows high performance in classifying happy and sad emotions, with minor misclassifications between neutral and angry.

## Future Improvements
- Experiment with **more advanced models** like Transformers for audio.  
- Use **data augmentation** to increase dataset diversity.  
- Integrate **real-time audio input** for live emotion detection.  
- Improve preprocessing with **noise reduction techniques**.

## License
MIT License – free to use.
