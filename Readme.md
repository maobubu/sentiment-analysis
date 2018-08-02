Huicheng Liu 20071349 17hl7@queensu.ca
1.Our code runs perfectly on ubuntu 16.04 and 17.10 operating system with python 3.6.

2.  Requirements has been listed in the "requirements.txt" file.(I'm afraid it will be a little bit troublesome to get the tensorflow GPU version configured. It might take a long time to run it with CPU) Type    pip3 install -r requirements.txt 

3.In the data folder, we only provide a small part of the meta data which contains about 37000 reviews for Instant video category. If you have more interest on the other dataset please follow the link“http://jmcauley.ucsd.edu/data/amazon/” and download whenever data you want. Perhaps you need to slightly modify the code. We shrink the pre-trained 50 Dimensional size word embedding into a pickle file called "glove_6B_50d.pickle" which only contains the word vectors for the word that has been used in the reviews for Instant video category.

4.Only the prediction results for the Instant video category has been proposed, "prediction_result_CNN.csv" represents the prediction result for CNN model, Wheread "prediction_result_lSTM.csv" represents the prediction result for LSTM model. Both results are based on the 50D word embedding and the Insant video category. Feel free to contact me if you want the other results.

5. To run the code, simply open the terminal and cd to the directory. Then type
    python3 main.py
The defalut model will be the CNN model. If you want to run the LSTM model, Open main.py and move to line 120,comment the CNN part and uncomment the LSTM part at line 125. That will make the code run on LSTM model. You will see two plots after running the code. LSTM model will cost more time then the CNN model. 
The runtime depends on your testing computer. It takes about 6 minutes for the CNN and 15 minutes for the LSTM to run on my laptop with GPU.

6.You don't have to run the "preprocess.py" and "word_vec.py" since they are the preprocessing step and we alrady run it for you. You will find out that "word_vec.py" will run perfectly but "preprocess.py" won't. That's because we didn't put the file "glove_6B_50d.txt" inside.(It's too big 178.9 MB , afte running it you can get the "glove_6B_50d.pickle" file mentioned above). You need to download the meta data from the link above and also download pretrained word embeddings from GLOVE "https://nlp.stanford.edu/projects/glove/". You will have to modify the file name and file path in "preprocess.py","word_vec.py" and "main.py". Contact me if you want to do this.

7. The final report, presentation and literature reviews are in the "Report, presentation and literature reviews" folder.

Feel free to contact me if you need any further help.
