import re
import collections
import itertools


def merge_data(headers, data):
    """
    merges data into a table
    is assumed that each cell is a length-2 tuple or list ['', '']
    """
    # The itertools function zip_longest() works with uneven lists
    transposed_table = list(itertools.zip_longest(*data, fillvalue=['', '']))
    firstrow = list(itertools.chain.from_iterable(transposed_table[0]))
    assert (len(headers) == len(firstrow)), "lists are of different size!"

    result = [headers]
    for i in transposed_table:
        # print(list(itertools.chain.from_iterable(i)))
        result.append(list(itertools.chain.from_iterable(i))) # flattens first list level
    return result

# lemario DRAE tomado de página no oficial. A 2010-10-17 eran 88449 palabras
with open('./lemario.txt', 'r') as f:
    raw_text = f.read()  # readlines()

# remove 'tildes' and uppercase letters
text = raw_text.lower().replace("á", "a").replace("é", "e").replace("í", "í").replace("ó", "o").replace("ú", "u")

# REGEX SECTION
r_syllableES = r"(?:ch|ll|rr|qu|[mnñvzsyjhxw]|[fpbtdkgc][lr]?|[lr])?" \
               r"(?:" \
               r"[iuü][eaoéáó][iyu]" \
               r"|[aá]h?[uú][aá]" \
               r"|[iuü]h?[eaoéáó]" \
               r"|[eaoéáó]h?[iyu]" \
               r"|[ií]h?[uú]" \
               r"|[uúü]h?[iíy]" \
               r"|[ieaouíéáóúü]" \
               r")" \
               r"(?:(?:" \
               r"(?:(?:n|m|r(?!r))s?(?![ieaouíéáóúü]))" \
               r"|(?:(?:[mnñvzsyjhxw]|l(?!l))(?![ieaouíéáóúü]))" \
               r"|(?:(?:[fpbtdkg]|c(?!h))(?![lr]?[ieaouíéáóúü]))" \
               r"))?"
# append one of this to regex above in order to differentiate final syllables from non final
final = r'$'
non_final = r'(?!$)'
# O N C
r_onset = "^[^aeiouáéíóúü]+"
r_nucleus = "[aeiouáéíóúü]+"
r_code = "[^aeiouáéíóúü]+$"



# hoy many SYLLABLES? SYLLABLES length
words_syllable_length = []
words = text.split("\n")
for i in words:
    syllables = re.findall(r_syllableES, i)
    words_syllable_length.append(len(syllables))
    # print(i, len(syllables))

words_syllable_length_freq = collections.Counter(words_syllable_length).items()


# EXTRACT SYLLABLES & final / non final
syllables = re.findall(r_syllableES, text, re.MULTILINE)
syllables_final = re.findall(r_syllableES + final, text, re.MULTILINE)
syllables_non_final = re.findall(r_syllableES + non_final, text, re.MULTILINE)

# SYLLABLE STRUCTURE
# ON ONC N NC & final / non final
# final
syllable_structure_final = []
for i in syllables_final:
    if re.search(r_onset, i):
        if re.search(r_code, i):
            syllable_structure_final.append('ONC')
        else:
            syllable_structure_final.append('ON')
    else:
        if re.search(r_code, i):
            syllable_structure_final.append('NC')
        else:
            syllable_structure_final.append('N')

syllable_structure_final_freq = collections.Counter(syllable_structure_final).items()

# non final
syllable_structure_non_final = []
for i in syllables_non_final:
    if re.search(r_onset, i):
        if re.search(r_code, i):
            syllable_structure_non_final.append('ONC')
        else:
            syllable_structure_non_final.append('ON')
    else:
        if re.search(r_code, i):
            syllable_structure_non_final.append('NC')
        else:
            syllable_structure_non_final.append('N')

syllable_structure_non_final_freq = collections.Counter(syllable_structure_non_final).items()

new_text = "\n".join(syllables)
# ONSETS
onsets = re.findall(r_onset, new_text, re.MULTILINE)
# NUCLEUS
nucleus = re.findall(r_nucleus, new_text, re.MULTILINE)
# CODES
codes = re.findall(r_code, new_text, re.MULTILINE)
codes_final = re.findall(r_code, "\n".join(syllables_final), re.MULTILINE)
codes_non_final = re.findall(r_code, "\n".join(syllables_non_final), re.MULTILINE)


# collections.Counter creates a frequency dictionary
# items() convert a dictionary into a list of tuples
onsets_freq = collections.Counter(onsets).items()
nucleus_freq = collections.Counter(nucleus).items()
codes_freq = collections.Counter(codes).items()
codes_final_freq = collections.Counter(codes_final).items()
codes_non_final_freq = collections.Counter(codes_non_final).items()


# prepare table output
table_temp = [words_syllable_length_freq, syllable_structure_non_final_freq, syllable_structure_final_freq,
              onsets_freq, nucleus_freq, codes_freq, codes_final_freq, codes_non_final_freq]
headings = ["words syllable length", "f", "syllable structure non final", "f", "syllable structure final", "f",
            "onset", "f", "nucleus", "f", "codes", "f", "codes_final", "f", "codes_non_final", "f"]
table = merge_data(headings, table_temp)

# export table to a 'tab separated values' file
with open('./output.tsv', 'w') as f:
    for row in table:
        f.write("\t".join(str(x) for x in row) + '\n')


