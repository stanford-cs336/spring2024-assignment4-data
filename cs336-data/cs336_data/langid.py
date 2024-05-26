import fasttext
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='`load_model` does not return WordVectorModel')

LANGID_PATH = "/home/shared/lid.176.bin"

def identify_language(text: str) -> tuple[str, float]:
    model = fasttext.load_model(LANGID_PATH)
    text = text.replace("\n", "")
    prediction = model.predict(text)
    label = prediction[0][0].replace("__label__", "")
    score = prediction[1][0]
    return label, score

if __name__=='__main__':
    languages = []
    scores = []
    for i in range(20):
        file_path = f'out/extract_warc{i+1}.txt'
        with open(file_path) as f:
            text = f.read()
            lan, scor = identify_language(text)
            languages.append(lan)
            scores.append(scor)
    print(languages)
    print(scores)