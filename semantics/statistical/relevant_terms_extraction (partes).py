import sys


def get_lemmas1(text):
    """
    Removes punctuation and function affixes; lexemes and derivational affixes remain
    :param text: text to tokenize
    :return: tokens
    """
    import spacy
    nlp = spacy.load("es_dep_news_trf")

    doc = nlp(text)
    a = [token.lemma_ for token in doc if token.pos_ not in ['PUNCT', 'SPACE']]
    sys.exit(0)


# get_lemmas1('Sin embargo pude verte en el 2010 por el ojo de buey')






# aa = """I remember watching one cartoon that is a retelling of the fairy tale by HC Andersen called "The Snow Queen". IIRC, it has a scene where Kai gets shards from the mirror in his eye and heart. I don't know if that version had the devil as the character who created the mirror or if, like the 1957 version, they had the Snow Queen as the mirror's owner in that version as well. Now, I remember only two scenes of note from that version. In one, after Kai gets the shard in his eye, there is a scene from his perspective. I believe, before that scene, he and Gerda are watering some flowers, and in this scene, his vision turns red and he sees one of their roses as being infested with worms, and so tramples on the rose, which upsets Gerda. In the other scene, Kai is abducted by the Snow Queen while sledding. He tried to detach his sled from the Snow Queen's carriage, but remarks that his efforts are futile since it's completely frozen to the carriage. Any idea as to what cartoon version of the Snow Queen this could be? This is definitely not the 1957 USSR version."""
# dd = aa.split('.')
# print(dd)

"""
Extraer información de un artículo de wikipedia,
Extraer la introducción de cada artículo, esto es, los párrafos previos a la tabla de contenidos
"""
import requests
from bs4 import BeautifulSoup
import re

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

    # children = soup.find_all('p')

    # In BeautifulSoup, saying “find” instead of “find_all” is going to only return the first instance of whatever you’re telling your code to find.
    toc = soup.find(id='toc')
    if toc:
        # for index, _ in enumerate(toc.previous_siblings()):
        #     print(_)
        # children = list(toc.previous_siblings("p"))
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

from nltk.tokenize import word_tokenize
import spacy


def get_lemmas(text):
    """
    Removes punctuation and function affixes; lexemes and derivational affixes remain
    :param text: text to tokenize
    :return: tokens
    """
    doc = nlp(text)
    # import explacy
    # explacy.print_parse_info(nlp, text)
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
        # print('\n' if hasattr(token, 'is_text_start') and token.is_text_start else '', token.dep_, '\t', token.pos_, '\t', token.text)
        if token.pos_ == 'SPACE':  # aparte
            lemmatized_sentences.append([])  # new sentence
            # print('\n')
        elif token.pos_ != 'PUNCT':
            lemma = token.lemma_
            lemmatized_sentences[-1].append(lemma)
            # print(token.lemma_)
    return lemmatized_sentences


# NOTA: existen diversos modelos para el NLP aparte del utilizado a continuación
nlp = spacy.load("es_dep_news_trf")
# fast = False
# if fast:
#     nlp = spacy.load("es_core_news_sm")
# else:  # más pesado:
#     nlp = spacy.load("es_dep_news_trf")

documents_lemmas = [get_lemmas(_) for _ in documents_text]
print(documents_lemmas)




from nltk.corpus import stopwords
stopwords = stopwords.words('spanish')
print(stopwords)


documents_filtered = [[token for token in document if not token.lower() in stopwords] for document in documents_lemmas]
print(documents_filtered)


def apuntes_borra(documents_tokens):
    import math
    """
    pip install xlsxwriter
    pip install xlwt
    """
    documents_frequencies = [collections.Counter(_) for _ in documents_tokens]
    df = pd.DataFrame(documents_frequencies)
    N = len(df)
    # df.to_csv('idfs-df1.tsv', sep='\t')  # , header=None, index=None)


    """
    count() method returns the number of non-NaN values
    in each column:
    df1.count()
    in each row.
    df1.count(axis=1)
    """
    # df['count'] = df.count(axis=1)
    rs = pd.DataFrame(df.count(axis=0)).transpose()
    print(rs)
    df1 = df.append(rs, ignore_index=True)
    # causes ERROR IndexError: arrays used as indices must be of integer (or boolean) type
    df1 = df1.transpose()
    # df1.to_csv('idfs-freq.tsv', sep='\t')  # , header=None, index=None)
    # print('count\n', df1)
    # df.drop(df.tail(1).index, inplace=True)

    # df.replace('', np.nan).count()
    for col in rs:
        rs[col] = math.log(N / rs[col])

    for col in df:
        counts = df[col].count()
        df[col] = df[col] * math.log(N / counts)
    # print('tf·idf\n', df)
    df2 = df.append(rs, ignore_index=True)
    df2 = df2.transpose()

    writer = pd.ExcelWriter('data_results/tf_ifg.xlsx', engine='xlsxwriter')
    # write each DataFrame to a specific sheet
    df1.to_excel(writer, sheet_name='frequencies')
    df2.to_excel(writer, sheet_name='tf·idf')
    writer.save()

    return
    for col in rs:
        rs[col] = math.log(N / rs[col])
    df2 = df.append(rs, ignore_index=True)
    print('idf\n', df2)
    transposed = df2.transpose()
    transposed.to_csv('idfs-idf.tsv', sep='\t')  # , header=None, index=None)
    # df2.to_csv('idfs-idf.tsv', sep='\t')  # , header=None, index=None)

    # df = df.set_index('time')
    # esto debería ser fila al final
    for col in df:
        print(col)
    for name, values in df.iteritems():
        print(values)
    for row, row_object in df.iterrows():
        # print('count:', counts[0][row])
        for col, value in enumerate(row_object):
            # print('\t', row, col, value)
            df.iloc[row, col] = math.log(N / counts[0][row])
            # df.iloc[row, col] = value * math.log(N / rs[0][row])
    return df


def pruebaidf():
    documents = [
        'I remember watching one cartoon that is a retelling of the fairy tale by HC Andersen called "The Snow Queen"',
        'IIRC, it has a scene where Kai gets shards from the mirror in his eye and heart',
        "I don't know if that version had the devil as the character who created the mirror or if, like the 1957 version, they had the Snow Queen as the mirror's owner in that version as well",
        'Now, I remember only two scenes of note from that version',
        'In one, after Kai gets the shard in his eye, there is a scene from his perspective',
        'I believe, before that scene, he and Gerda are watering some flowers, and in this scene, his vision turns red and he sees one of their roses as being infested with worms, and so tramples on the rose, which upsets Gerda',
        'In the other scene, Kai is abducted by the Snow Queen while sledding',
        "He tried to detach his sled from the Snow Queen's carriage, but remarks that his efforts are futile since it's completely frozen to the carriage",
        'Any idea as to what cartoon version of the Snow Queen this could be? This is definitely not the 1957 USSR version',
    ]

    documents_frequencies = [
        {'points': 5, 'time': 13, 'year': 1, 'hell': 21},
        {'points': 5, 'time': 13, 'year': 1},
        {'points': 7, 'time': 9, 'year': 2},
        {'points': 9, 'time': 7},
        {'points': 1},
    ]

    documents = [
        'la casa es casa grande',
        'la moto no es nueva',
        'no he visto la serie que me comentas',
    ]
    documents_tokens = [collections.Counter(document.split()) for document in documents]
    documents_frequencies = [collections.Counter(_) for _ in documents_tokens]

    df = pd.DataFrame(documents_frequencies)
    N = len(df)

    for row, row_object in df.iterrows():
        print('\n', row)
        print(row_object[row_object.values > 0])
        print(row_object[row_object.values > 0].sort_values(ascending=False))
        sd = row_object[row_object.values > 0].to_dict()
        print(sd)
        # print(row_object > 0)

    threshold = 0
    result = [row_object[row_object.values > threshold].to_dict()
              for row, row_object in df.iterrows()]
    print('\n', len(result), result)

    rs = pd.DataFrame(df.count(axis=0)).transpose()
    df1 = df.append(rs, ignore_index=True)
    df1 = df1.transpose()
    for col in rs:
        rs[col] = math.log(N / rs[col])

    print(df)

    for col in df:
        counts = df[col].count()
        df[col] = df[col] * math.log(N / counts)

    print(df)
    results = df.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=2)
    print(results)

    df2 = df.append(rs, ignore_index=True)
    df2 = df2.transpose()

    sys.exit(0)



"""
pip install xlsxwriter
pip install xlwt
"""
import math
import collections
import pandas as pd


def get_tfidf(documents_tokens, max_results=1000, threshold=0, save_as=None):
    import math

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


# pruebaidf()
# from tfidf import get_idf


df, tfidf_weights = get_tfidf(documents_filtered, max_results=1000, threshold=0, save_as='relevant_terms_extraction')
print(df)
print(tfidf_weights)

sys.exit(0)

