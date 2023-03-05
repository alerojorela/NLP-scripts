"""
pip install xlsxwriter
pip install xlwt
"""
import requests
from bs4 import BeautifulSoup
import re
import spacy
from nltk.corpus import stopwords
import math
import collections
import pandas as pd


def extract_wikipedia_article(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, features='lxml')

    content_text = soup.find(id='mw-content-text')
    children = content_text.find_all('p')
    text = '\n'.join(_.get_text() for _ in children)
    text = re.sub(r'\[\d+\]|​', '', text)
    return text


def extract_wikipedia_intro(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, features='lxml')

    toc = soup.find(id='toc')
    if toc:
        children = []
        node = toc
        while node.find_previous_sibling('p'):
            node = node.find_previous_sibling('p')
            children.append(node)
        children = children[::-1]
    else:  # no toc found -> extract all paragraphs
        children = soup.findChildren("p", recursive=False)

    text = '\n'.join(_.get_text() for _ in children)
    text = re.sub(r'\[\d+\]|​| ', '', text)
    return text


def get_lemmas(text):
    """
    Removes punctuation and function affixes; lexemes and derivational affixes remain
    :param text: text to tokenize
    :return: tokens
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ not in ['PUNCT', 'SPACE']]


def get_lemmatized_sentences(text):
    """
    Removes punctuation and function affixes; lexemes and derivational affixes remain

    :param text: text to tokenize
    :return: tokenized sentences
    """
    doc = nlp(text)
    lemmatized_sentences = [[]]
    for token in doc:
        if token.pos_ == 'SPACE':  # aparte
            lemmatized_sentences.append([])  # new sentence
            # print('\n')
        elif token.pos_ != 'PUNCT':
            lemma = token.lemma_
            lemmatized_sentences[-1].append(lemma)
            # print(token.lemma_)
    return lemmatized_sentences


def tfidf(documents_tokens, max_results=1000, threshold=0):
    documents_frequencies = [collections.Counter(_) for _ in documents_tokens]
    df = pd.DataFrame(documents_frequencies)
    length = len(df)

    for col in df:
        counts = df[col].count()
        df[col] = df[col] * math.log(length / counts)

    tfidf_weights = [row_object[row_object.values > threshold].sort_values(ascending=False).head(max_results).to_dict()
                     for row, row_object in df.iterrows()]
    [print(index, len(_), _) for index, _ in enumerate(tfidf_weights)]

    return df, tfidf_weights


def export_tfidf(documents_tokens, max_results=1000, threshold=0, save_as=None):
    documents_frequencies = [collections.Counter(_) for _ in documents_tokens]
    df = pd.DataFrame(documents_frequencies)
    length = len(df)

    rs = pd.DataFrame(df.count(axis=0)).transpose()
    export_df1 = df.append(rs, ignore_index=True).transpose()

    for col in rs:
        rs[col] = math.log(length / rs[col])

    for col in df:
        counts = df[col].count()
        df[col] = df[col] * math.log(length / counts)

    export_df2 = df.append(rs, ignore_index=True).transpose()

    tfidf_weights = [row_object[row_object.values > threshold].sort_values(ascending=False).head(max_results).to_dict()
                     for row, row_object in df.iterrows()]
    [print(index, len(_), _) for index, _ in enumerate(tfidf_weights)]

    # https://stackoverflow.com/questions/34518634/finding-highest-values-in-each-row-in-a-data-frame-for-python
    highest_keys = df.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=max_results)
    if save_as:
        writer = pd.ExcelWriter(save_as, engine='xlsxwriter')
        # write each DataFrame to a specific sheet
        export_df1.to_excel(writer, sheet_name='frequencies')
        export_df2.to_excel(writer, sheet_name='tf·idf')
        highest_keys.to_excel(writer, sheet_name='highest_keys')
        writer.save()

    return df, tfidf_weights


documents_urls = [
    'https://es.wikipedia.org/wiki/%C3%81tomo',
    'https://es.wikipedia.org/wiki/C%C3%A9lula',
    'https://es.wikipedia.org/wiki/Cerebro',
    'https://es.wikipedia.org/wiki/Unidad_central_de_procesamiento',
    'https://es.wikipedia.org/wiki/Ciudad',
    'https://es.wikipedia.org/wiki/Sol',
]

documents_text = [extract_wikipedia_article(_) for _ in documents_urls]
print(documents_text)

# NOTA: existen diversos modelos para el NLP aparte del utilizado a continuación
nlp = spacy.load("es_dep_news_trf")
documents_lemmas = [get_lemmas(_) for _ in documents_text]
print(documents_lemmas)

stopwords = stopwords.words('spanish')
print(stopwords)

documents_filtered = [[token for token in document if not token.lower() in stopwords] for document in documents_lemmas]
print(documents_filtered)

df, tfidf_weights = export_tfidf(documents_filtered, max_results=1000, threshold=0, save_as='relevant_terms_extraction.xlsx')
print(df)
print(tfidf_weights)
