
import os
import random
import string
from typing import Optional


CREATED_FILE_NAMES = []

def remove_created_files():
    while len(CREATED_FILE_NAMES) != 0:
        f = CREATED_FILE_NAMES.pop()
        os.remove(f)

def generate_file_name(postfix):
    def gen_name(postfix: Optional[str], length=8):
        if postfix is None:
            postfix = ''
        choices = string.ascii_letters + string.digits
        return ''.join([random.choice(choices) for _ in range(length)]) + postfix

        
    filename = gen_name(postfix)
    while os.path.exists(filename):
        filename = gen_name(postfix)
    return filename