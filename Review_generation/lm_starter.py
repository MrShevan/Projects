import argparse, re, os
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import pandas as pd
from time import time, sleep
import random

from lm_model import Model
from lm_dataset import Dataset

'''
This class downloads all school compositions from litra.ru on a given topic 
By default: Сочинения по произведению "Евгений Онегин" (Пушкин А.С.)
'''
class Litra_parser:
    def __init__(self, general_url='http://www.litra.ru/composition/work/woid/00028601184773070301/'):
        self.general_url = general_url
        self.prefix = 'http://www.litra.ru'
        self.urles = []
        self.sess = None
    

    def take_urles(self):
        self.sess = requests.Session()
        g = self.sess.get(self.general_url)
        b = BeautifulSoup(g.text, "html.parser")
        element = b.find('div', {'id': 'child_id2'})
        self.urles = re.findall('<a href=[^>]+', str(element))
        self.urles = [self.prefix + x.replace('<a href="', '').replace('"', '') for x in self.urles]


    def take_compositions(self, save_on_local=True):
        texts = []
        print('Downloading compositions...')
        os.mkdir('reviews')
        for i, url in tqdm(enumerate(self.urles), total = len(self.urles)):
            g = self.sess.get(url)
            b = BeautifulSoup(g.text, "html.parser")
            tags_list = b.find_all('p', {'align':'justify'})
    
            # composition preprocessing
            text = re.sub('[^а-яА-Я0-9a-zA-z;?!:,. -/s]', '', tags_list[0].text)
            text = re.sub('[-/s;:""()]', ' ', text)
            text = text.replace(',', ' , ')
            text = text.replace('.', ' . ')
            text = text.replace('.', ' . ')
            text = text.replace('?', ' ? ')
            text = text.replace('!', ' ! ')
            text = re.sub('[ ]+', ' ', text)
            text = text.lower()
            text = text.strip(' ')

            texts.append(text)

            if save_on_local == True:
                with open('reviews/review_' + ('%03d' % i) + '.txt', 'w') as f:
                    f.write(text)
        
            sleep(1)
        print('Done!')
        return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=5, type=int, help='epoch size')
    parser.add_argument('--download_dataset', default=1, type=int)
    parser.add_argument('--starting_word', default=None, type=str)
    args = parser.parse_args()

    '''Сочинения по произведению "Евгений Онегин" (Пушкин А.С.)'''
    general_url = 'http://www.litra.ru/composition/work/woid/00028601184773070301/'

    # Data scrapping:
    if args.download_dataset == 1:
        web_scrapper = Litra_parser(general_url)
        web_scrapper.take_urles()
        texts = web_scrapper.take_compositions()
    else:
        print('Reading...')
        texts = []
        for i in tqdm(range(306), total=306):
            with open('reviews/review_' + ('%03d' % i) + '.txt', 'r') as file:
                texts.append(file.read())
        print('Done!')

    text = ' '.join(texts)
    text_words = text.split(' ')
    vocab = set(text_words)

    # Prepare dataset:
    print('Words in texts: %d' % len(text_words))
    print('Unique words in texts: %d' % len(set(text_words)))

    df = Dataset(seq_length=15, BATCH_SIZE = 40)
    df.make(text_words)
    print('Dataset generated')

    # Build and train model:
    model_params = {
        'df' : df,
        'vocab_size' : len(df.vocab),
        'embedding_dim' : 300,
        'units' : 1024,
        'rate' : 0.5,
        'EPOCHS' : args.epoch
    }

    model = Model(**model_params)
    model.train()

    # Generate text:
    if not args.starting_word:
        start_string = random.choice(list(vocab))
    else:
        start_string = args.starting_word

    num_generate = 25
    model.generate_text(start_string, num_generate)


if __name__ == '__main__':
    main()
