import fasttext

NSFW_PATH = "/home/shared/dolma-jigsaw-fasttext-bigrams-nsfw.bin"
HATESPEECH_PATH = "/home/shared/dolma-jigsaw-fasttext-bigrams-hatespeech.bin"

def classify_nsfw(text: str) -> tuple[str, float]:
    model = fasttext.load_model(NSFW_PATH)
    text = text.replace("\n", "")
    prediction = model.predict(text)
    label = prediction[0][0].replace("__label__", "")
    score = prediction[1][0]
    return label, score

def classify_toxic_speech(text: str) -> tuple[str, float]:
    model = fasttext.load_model(HATESPEECH_PATH)
    text = text.replace("\n", "")
    prediction = model.predict(text)
    label = prediction[0][0].replace("__label__", "")
    score = prediction[1][0]
    return label, score

if __name__=='__main__':
    for i in range(1, 10):
        with open(f'out/extract_warc{i}.txt', 'r') as f:
            text = f.read()
            label1, score1 = classify_nsfw(text)
            label2, score2 = classify_toxic_speech(text)
            print(label2)