# import modules
import csv
import string
import nltk
from nltk.tokenize import word_tokenize

# import list of emotion words/lemmas and create dictionary with words to count
file = open('Topic emotion lemmas_100.csv', 'r', encoding='utf-8-sig')
emotions = list(csv.reader(file, delimiter=","))
file.close()
emotions = [row[0] for row in emotions]

# run through data (topic distribution matrix output) and count emotion words/lemmas
with (open('Topic Distribution Matrix.csv', 'r', encoding='utf-8-sig') as fr,
      open('Topic Distribution Matrix_counts.csv', 'w', encoding='utf-8-sig', newline='') as fw):
    reader = csv.reader(fr)
    writer = csv.writer(fw)

    next(fr)
    header = ['CID', 'beep', 'dateTime', 'rawText', 'inputText', 'matched']
    header.extend(emotions)
    writer.writerow(header)

    for row in reader:
        CID = row[0]
        beep = row[2]
        dateTime = row[3]
        rawText = row[10] # retain raw text in output, if desired
        inputText = row[15] # specify text pre-masking
        matched = row[17]
        text = inputText
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = [token for token in word_tokenize(text)]
        emotions_dictionary = dict.fromkeys(emotions, 0)
        for word in words:
            if word.lower() in emotions_dictionary:
                emotions_dictionary[word.lower()] += 1
        emotion_counts = list(emotions_dictionary.values())
        to_write = [CID, beep, dateTime, rawText, inputText, matched]
        to_write.extend(emotion_counts)
        writer.writerow(to_write)