from typing import Tuple
import re

def mask_emails(text: str) -> Tuple[str, int]:
    replace_text = '|||EMAIL_ADDRESS|||'
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    masked_text, count = re.subn(email_pattern, replace_text, text)
    return masked_text, count

def mask_phone_numbers(text: str) -> Tuple[str, int]:
    replace_text = '|||PHONE_NUMBER|||'
    phone_pattern = r'\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}'
    masked_text, count = re.subn(phone_pattern, replace_text, text)
    return masked_text, count

def mask_ips(text: str) -> Tuple[str, int]:
    replace_text = '|||IP_ADDRESS|||'
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    masked_text, count = re.subn(ip_pattern, replace_text, text)
    return masked_text, count

if __name__=='__main__':
    for i in range(1, 4):
        with open(f'out/extract_warc{i}.txt', 'r') as f:
            text = f.read()
            masked_text, count = mask_emails(text)
            masked_text, count = mask_phone_numbers(masked_text)
            masked_text, count = mask_ips(masked_text)
            print(masked_text)