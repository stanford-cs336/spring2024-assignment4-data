from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree
from resiliparse.parse.encoding import detect_encoding
from xopen import xopen
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
import sys

def read_warc_file(file_path: str):
    with xopen(file_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            yield record.reader.read()

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)
    html_bytes = HTMLTree.parse_from_bytes(html_bytes, encoding)
    return extract_plain_text(html_bytes, encoding)


if __name__=='__main__':
    input_path = '/home/shared/CC-examples/example.warc.gz'
    records = read_warc_file(input_path)
    
    # extract first 20 record and save as txt
    count = 0
    for record in tqdm(records):
        if count > 20:
            break
        else:
            count += 1
            with open(f'out/extract_warc{count}.txt', 'w') as f:
                text = extract_text_from_html_bytes(record)
                f.write(text)
    print('Done')
                
