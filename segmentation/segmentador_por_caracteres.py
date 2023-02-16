#! /usr/bin/python3
# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
import sys

regexes = {
    'segmenta_por_blancos': [
        r'[^ \t]+',  # r'(?:[^ \t]+)'
        r'|[ \t]+',
        r'(?:[ \t]+)?'
    ],

    'segmenta_por_parrafos': [
        r'[^\n\r]+',
        r'|[\n\r]+',
        r'(?:[\n\r]+)?'
    ],

    'segmenta_por_puntos': [
        r'(?:[^.]+' \
        r'|(?<=\d)\.(?=\d)' \
        r'|(?<=\[)\.(?=\.\.\])' \
        r'|(?<=\[\.)\.(?=\.\])' \
        r'|(?<=\[\.\.)\.(?=\])' \
        r'|(?<=\.[A-ZÑ])\.' \
        r'|\.(?=[A-ZÑ]\.)' \
        r'|(?<=\.[A-ZÑ]{2,2})\.' \
        r'|\.(?=[A-ZÑ]{2,2}\.))+',

        r'|\.(?:\.\.)?',
        r'(?:\.(?:\.\.)?)?'
    ],

    'segmenta_por_dos_puntos': [
        r'(?:[^:]+|(?<=\d):(?=\d))+',
        r'|:',
        r':?'
    ],
    'segmenta_por_comas': [
        r'(?:[^,]+|(?<=\d),(?=\d))+',
        r'|,',
        r',?'
    ],
    'segmenta_por_barras': [
        r'(?:[^\/\\]+|(?<=\d)[\/\\](?=\d))+',
        r'|[\/\\]',
        r'(?:[\/\\]+)?'
    ],

    'segmenta_por_guiones': [
        r'(?:[^-]+|-(?=\d)|(?<=\w)-(?=\w))+',
        r'|-',
        r'-?'
    ],
    'segmenta_por_apostrofo': [
        r"(?:[^'´’]+|(?<=\w)['´’](?=\w))+",
        r'|[\'´’]',
        r'(?:[\'´’]+)?'
    ],
    'segmenta_por_varios': [
        r'(?:[^"“”«»<>@%&#()\[\]{}‘`•·]+' \
        r'|\[(?=\.\.\.\])' \
        r'|(?<=\[\.\.\.)\]' \
        r')+',
        # [ ...]

        r'|["“«”»<>@%&#()\[\]{}‘`•·]',
        r'(?:["“«”»<>@%&#()\[\]{}‘`•·]+)?'
    ],

    'segmenta_por_no_ambiguos': [
        r'[^¿?¡!…]+',
        r'[^¿?¡!…]+|[¿?¡!…]',
        r'(?:(?:[¿¡]+)?(?:(?:[^¿?¡!…])+)?(?:[?!…]+)?)'
    ]
}


def procesa_expresion(strings_a_segmentar, nombre_regex,
                      elimina_separadores=False, separadores_aparte=True, elimina_espacios=True):
    """
    :param strings_a_segmentar: La entrada, como lista de strings.
    :type strings_a_segmentar: List[str]
    :param regex:
    :type regex: str
    :param elimina_separadores: Si está a True, se elimina el separador. Si está a False, el separador se
        mantiene (aparecerá como token independiente o a final de frase según el valor de separadores_aparte).
    :type elimina_separadores: bool
    :param separadores_aparte: Si está a True, los separadores aparecen como tokens aparte. Si no, aparecen
        al final de la frase.
    :type separadores_aparte: bool
    :param elimina_espacios:
    :type elimina_espacios: bool
    :return: Una lista de strings.
    :rtype: List[str]
    """
    if nombre_regex == 'segmenta_por_no_ambiguos':
        if elimina_separadores:
            regex = r'[^¿?¡!…]+'
        elif separadores_aparte:
            regex = r'[^¿?¡!…]+|[¿?¡!…]'
        else:
            regex = r'(?:(?:[¿¡]+)?(?:(?:[^¿?¡!…])+)?(?:[?!…]+)?)'
    else:
        entry = regexes[nombre_regex]
        if elimina_separadores:
            regex = entry[0]
        elif separadores_aparte:
            regex = entry[0] + entry[1]
        else:
            regex = entry[0] + entry[2]

    # Se crea el segmentador usando la expresión regular que hayamos creado.
    segmentador = RegexpTokenizer(regex)
    # Se obtienen los resultados: para cada string de la lista de entrada, se crea una lista de strings (que
    # habitualmente será un único string salvo que incluya un ':' que separe frases), y todos esos strings
    # se devuelven en una lista. Se devuelven al menos tantos strings como tenga la lista de entrada.
    if elimina_espacios:
        return [string.strip()
                for string_inicial in strings_a_segmentar
                for string in segmentador.tokenize(string_inicial)
                if string.strip()]
    else:
        return [string
                for string_inicial in strings_a_segmentar
                for string in segmentador.tokenize(string_inicial)]


def segmenta_por_tokens(strings_a_segmentar, elimina_separadores=False):
    """
    Tomamos una lista de strings (un texto previamente segmentado como lista de strings, o un único texto pero
    dentro de una lista de un único elemento) y devolvemos otra lista en la que cada string de la lista de
    entrada se convierte en una lista de tokens (es decir, se devuelve una lista de listas de strings).
    Los tokens que sean signos de puntuación (en general, separadores) se eliminan o no según el parámetro.
    El uso general de esta función es la de tomar una lista de strings que representan a frases (que incluyen
    los separadores al inicio/final) y devolver una lista de listas de tokens, donde cada lista de tokens
    incluidos en la lista de nivel superior representa una frase segmentado en tokens.

    :param strings_a_segmentar: La entrada, como lista de strings.
    :type strings_a_segmentar: List[str]
    :param elimina_separadores: Si está a True, se eliminan los separadores. Si está a False, los separadores
        se mantienen (aparecerán como tokens independientes).
    :type elimina_separadores: bool
    :return: Una lista de strings.
    :rtype: List[List[str]]
    """
    # copia de lista
    strings_a_segmentar_temp = strings_a_segmentar.copy()

    for n in range(0, len(strings_a_segmentar_temp)):
        # Hay que prestar atención a que el argumento de cada función sea una lista si no queremos que nos liste los caracteres individuales
        frase = [strings_a_segmentar_temp[n]]

        # funciones que segmentan frases pero también tokens
        frase = procesa_expresion(frase, 'segmenta_por_parrafos', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_dos_puntos', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_puntos', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_no_ambiguos', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)

        # funciones que segmentan tokens
        frase = procesa_expresion(frase, 'segmenta_por_comas', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_barras', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_guiones', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_apostrofo', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_varios', elimina_separadores=elimina_separadores,
                                  separadores_aparte=True)
        frase = procesa_expresion(frase, 'segmenta_por_blancos', elimina_separadores=True, elimina_espacios=False)
        strings_a_segmentar_temp[n] = frase

    return strings_a_segmentar_temp


def segmenta_por_frases(strings_a_segmentar, elimina_separadores=False, segmenta_tokens_de_frase=False):
    """
    Tomamos una lista de strings (un texto previamente segmentado como lista de strings, o un único texto pero
    dentro de una lista de un único elemento) y dependiendo del parámetro segmenta_tokens_de_frase se devuelve
    una de las dos siguientes cosas:
    - Si dicho parámetro es False, devolvemos otra lista de strings en la que los strings son frases. Los
      separadores de las frases NO SE SEPARAN y dicho caracter (o caracteres) van junto con el resto de
      caracteres que forman la frase.
    - Si el parámetro de segmenta_tokens_de_frases es True, esas frases se tokenizan a su vez de forma que
      cada frase se convierte en una lista de tokens (con lo que se devuelve no una lista de strings -una
      lista de frases donde cada frase está representada por un string- sino una lista donde cada elemento es
      una lista de strings -con lo que se devuelve una lista de listas de tokens-).
    Obviamente, al tokenizar así, los segmentadores aparecen como tokens aparte (y como parte de la lista de
    tokens que representa a la frase).

    :param strings_a_segmentar: La entrada, como lista de strings.
    :type strings_a_segmentar: List[str]
    :param elimina_separadores: Si está a True, se elimina el separador. Si está a False, el separador se
        mantiene. En cualquier caso, los separadores blancos siempre se eliminan.
    :type elimina_separadores: bool
    :param segmenta_tokens_de_frase: Si este parámetro está a True, una vez que se haya segmentado en frases
        dichas frases se subdividen a su vez en tokens. Si los separadores no se han eliminado, entonces
        aparecerán como tokens independientes. Así pues, si este parámetro es False se devuelve una lista de
        strings, y si está a True se devuelve una lista de listas de strings.
    :type segmenta_tokens_de_frase: bool
    :return: Una lista de strings o una lista de listas de strings (según el valor de
        segmenta_tokens_de_frase).
    :rtype: List[str] o List[List[str]]
    """
    # funciones que segmentan frases pero también tokens
    strings_a_segmentar_temp = procesa_expresion(strings_a_segmentar, 'segmenta_por_parrafos',
                                                 elimina_separadores=elimina_separadores,
                                                 separadores_aparte=False)
    strings_a_segmentar_temp = procesa_expresion(strings_a_segmentar_temp, 'segmenta_por_dos_puntos',
                                                 elimina_separadores=elimina_separadores,
                                                 separadores_aparte=False)
    strings_a_segmentar_temp = procesa_expresion(strings_a_segmentar_temp, 'segmenta_por_puntos',
                                                 elimina_separadores=elimina_separadores,
                                                 separadores_aparte=False)
    strings_a_segmentar_temp = procesa_expresion(strings_a_segmentar_temp, 'segmenta_por_no_ambiguos',
                                                 elimina_separadores=elimina_separadores,
                                                 separadores_aparte=False)

    if segmenta_tokens_de_frase:
        strings_a_segmentar_temp = segmenta_por_tokens(strings_a_segmentar_temp, elimina_separadores)

    return strings_a_segmentar_temp


# desde consola
# python segmentador.py
if __name__ == "__main__":

    texto = """El túmulo alargado Coldrum (en inglés Coldrum Long Barrow), también conocido como piedras Coldrum (Coldrum Stones) o piedras Adscombe (Adscombe Stones), es un túmulo alargado con cámara ubicado cerca del pueblo de Trottiscliffe del condado de Kent, en el sudeste de Inglaterra. Probablemente construido en el cuarto milenio antes de Cristo, durante el período Neolítico inicial de Gran Bretaña, se encuentra en estado de ruina. """
    if len(sys.argv) == 2:
        if sys.argv[1]:
            texto = sys.argv[1]
    else:
        print("Ejemplo: ")

    resultado = segmenta_por_frases([texto], segmenta_tokens_de_frase=True)
    [print(token) for token in resultado]
