# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:51:51 2019

@author: LG-PC
"""

'''
from gtts import gTTS
import os
text = "안녕하세요"

tts = gTTS(text=text, lang='ko')
tts.save("test.mp3")
os.system("test.mp3")
'''

import hgtk
import pytesseract as pt

MATCH_H2B_CHO = {
    u'ㄱ': [[0,0,0,1,0,0]],
    u'ㄴ': [[1,0,0,1,0,0]],
    u'ㄷ': [[0,1,0,1,0,0]],
    u'ㄹ': [[0,0,0,0,1,0]],
    u'ㅁ': [[1,0,0,0,1,0]],
    u'ㅂ': [[0,0,0,1,1,0]],
    u'ㅅ': [[0,0,0,0,0,1]],
    u'ㅇ': [[1,1,0,1,1,0]],
    u'ㅈ': [[0,0,0,1,0,1]],
    u'ㅊ': [[0,0,0,0,1,1]],
    u'ㅋ': [[1,1,0,1,0,0]],
    u'ㅌ': [[1,1,0,0,1,0]],
    u'ㅍ': [[1,0,0,1,1,0]],
    u'ㅎ': [[0,1,0,1,1,0]],

    u'ㄲ': [[0,0,0,0,0,1], [0,0,0,1,0,0]],
    u'ㄸ': [[0,0,0,0,0,1], [0,1,0,1,0,0]],
    u'ㅃ': [[0,0,0,0,0,1], [0,0,0,1,1,0]],
    u'ㅆ': [[0,0,0,0,0,1], [0,0,0,0,0,1]],
    u'ㅉ': [[0,0,0,0,0,1], [0,0,0,1,0,1]],
}

MATCH_H2B_JOONG = {
    u'ㅏ': [[1,1,0,0,0,1]],
    u'ㅑ': [[0,0,1,1,1,0]],
    u'ㅓ': [[0,1,1,1,0,0]],
    u'ㅕ': [[1,0,0,0,1,1]],
    u'ㅗ': [[1,0,1,0,0,1]],
    u'ㅛ': [[0,0,1,1,0,1]],
    u'ㅜ': [[1,0,1,1,0,0]],
    u'ㅠ': [[1,0,0,1,0,1]],
    u'ㅡ': [[0,1,0,1,0,1]],
    u'ㅣ': [[1,0,1,0,1,0]],
    u'ㅐ': [[1,1,1,0,1,0]],
    u'ㅔ': [[1,0,1,1,1,0]],
    u'ㅒ': [[0,0,1,1,1,0], [1,1,1,0,1,0]],
    u'ㅖ': [[0,0,1,1,0,0]],
    u'ㅘ': [[1,1,1,0,0,1]],
    u'ㅙ': [[1,1,1,0,0,1], [1,1,1,0,1,0]],
    u'ㅚ': [[1,0,1,1,1,1]],
    u'ㅝ': [[1,1,1,1,0,0]],
    u'ㅞ': [[1,1,1,1,0,0], [1,1,1,0,1,0]],
    u'ㅟ': [[1,0,1,1,0,0], [1,1,1,0,1,0]],
    u'ㅢ': [[0,1,0,1,1,1]],
}

MATCH_H2B_JONG = {
    u'ㄱ': [[1,0,0,0,0,0]],
    u'ㄴ': [[0,1,0,0,1,0]],
    u'ㄷ': [[0,0,1,0,1,0]],
    u'ㄹ': [[0,1,0,0,0,0]],
    u'ㅁ': [[0,1,0,0,0,1]],
    u'ㅂ': [[1,1,0,0,0,0]],
    u'ㅅ': [[0,0,1,0,0,0]],
    u'ㅇ': [[0,1,1,0,1,1]],
    u'ㅈ': [[1,0,1,0,0,0]],
    u'ㅊ': [[0,1,1,0,0,0]],
    u'ㅋ': [[0,1,1,0,1,0]],
    u'ㅌ': [[0,1,1,0,0,1]],
    u'ㅍ': [[0,1,0,0,1,1]],
    u'ㅎ': [[0,0,1,0,1,1]],

    u'ㄲ': [[1,0,0,0,0,0], [1,0,0,0,0,0]],
    u'ㄳ': [[1,0,0,0,0,0], [0,0,1,0,0,0]],
    u'ㄵ': [[0,1,0,0,1,0], [1,0,1,0,0,0]],
    u'ㄶ': [[0,1,0,0,1,0], [0,0,1,0,1,1]],
    u'ㄺ': [[0,1,0,0,0,0], [1,0,0,0,0,0]],
    u'ㄻ': [[0,1,0,0,0,0], [0,1,0,0,0,1]],
    u'ㄼ': [[0,1,0,0,0,0], [1,1,0,0,0,0]],
    u'ㄽ': [[0,1,0,0,0,0], [0,0,1,0,0,0]],
    u'ㄾ': [[0,1,0,0,0,0], [0,1,1,0,0,1]],
    u'ㄿ': [[0,1,0,0,0,0], [0,1,0,0,1,1]],
    u'ㅀ': [[0,1,0,0,0,0], [0,0,1,0,1,1]],
    u'ㅄ': [[1,1,0,0,0,0], [0,0,1,0,0,0]],
    u'ㅆ': [[0,0,1,1,0,0]],
}

MATCH_H2B_ALPHABET = {
    'a': [[1,0,0,0,0,0]],
    'b': [[1,1,0,0,0,0]],
    'c': [[1,0,0,1,0,0]],
    'd': [[1,0,0,1,1,0]],
    'e': [[1,0,0,0,1,0]],
    'f': [[1,1,0,1,0,0]],
    'g': [[1,1,0,1,1,0]],
    'h': [[1,1,0,0,1,0]],
    'i': [[0,1,0,1,0,0]],
    'j': [[0,1,0,1,1,0]],
    'k': [[1,0,1,0,0,0]],
    'l': [[1,1,1,0,0,0]],
    'm': [[1,0,1,1,0,0]],
    'n': [[1,0,1,1,1,0]],
    'o': [[1,0,1,0,1,0]],
    'p': [[1,1,1,1,0,0]],
    'q': [[1,1,1,1,1,0]],
    'r': [[1,1,1,0,1,0]],
    's': [[0,1,1,1,0,0]],
    't': [[0,1,1,1,1,0]],
    'u': [[1,0,1,0,0,1]],
    'v': [[1,1,1,0,0,1]],
    'w': [[0,1,1,1,1,1]],
    'x': [[1,0,1,1,0,1]],
    'y': [[1,0,1,1,1,1]],
    'z': [[1,0,1,0,1,1]],

    'A': [[0,0,0,0,0,1], [1,0,0,0,0,0]],
    'B': [[0,0,0,0,0,1], [1,1,0,0,0,0]],
    'C': [[0,0,0,0,0,1], [1,0,0,1,0,0]],
    'D': [[0,0,0,0,0,1], [1,0,0,1,1,0]],
    'E': [[0,0,0,0,0,1], [1,0,0,0,1,0]],
    'F': [[0,0,0,0,0,1], [1,1,0,1,0,0]],
    'G': [[0,0,0,0,0,1], [1,1,0,1,1,0]],
    'H': [[0,0,0,0,0,1], [1,1,0,0,1,0]],
    'I': [[0,0,0,0,0,1], [0,1,0,1,0,0]],
    'J': [[0,0,0,0,0,1], [0,1,0,1,1,0]],
    'K': [[0,0,0,0,0,1], [1,0,1,0,0,0]],
    'L': [[0,0,0,0,0,1], [1,1,1,0,0,0]],
    'M': [[0,0,0,0,0,1], [1,0,1,1,0,0]],
    'N': [[0,0,0,0,0,1], [1,0,1,1,1,0]],
    'O': [[0,0,0,0,0,1], [1,0,1,0,1,0]],
    'P': [[0,0,0,0,0,1], [1,1,1,1,0,0]],
    'Q': [[0,0,0,0,0,1], [1,1,1,1,1,0]],
    'R': [[0,0,0,0,0,1], [1,1,1,0,1,0]],
    'S': [[0,0,0,0,0,1], [0,1,1,1,0,0]],
    'T': [[0,0,0,0,0,1], [0,1,1,1,1,0]],
    'U': [[0,0,0,0,0,1], [1,0,1,0,0,1]],
    'V': [[0,0,0,0,0,1], [1,1,1,0,0,1]],
    'W': [[0,0,0,0,0,1], [0,1,1,1,1,1]],
    'X': [[0,0,0,0,0,1], [1,0,1,1,0,1]],
    'Y': [[0,0,0,0,0,1], [1,0,1,1,1,1]],
    'Z': [[0,0,0,0,0,1], [1,0,1,0,1,1]],

    '1': [[0,0,1,1,1,1], [1,0,0,0,0,0]],
    '2': [[0,0,1,1,1,1], [1,1,0,0,0,0]],
    '3': [[0,0,1,1,1,1], [1,0,0,1,0,0]],
    '4': [[0,0,1,1,1,1], [1,0,0,1,1,0]],
    '5': [[0,0,1,1,1,1], [1,0,0,0,1,0]],
    '6': [[0,0,1,1,1,1], [1,1,0,1,0,0]],
    '7': [[0,0,1,1,1,1], [1,1,0,1,1,0]],
    '8': [[0,0,1,1,1,1], [1,1,0,0,1,0]],
    '9': [[0,0,1,1,1,1], [0,1,0,1,0,0]],
    '0': [[0,0,1,1,1,1], [0,1,0,1,1,0]],

    ',': [[0,1,0,0,0,0]],
    '.': [[0,1,0,0,1,1]],
    '-': [[0,1,0,0,1,0]],
    '?': [[0,1,1,0,0,1]],
    '_': [[0,0,1,0,0,1]],
    '!': [[0,1,1,0,1,0]],
}


def letter(hangul_letter):
    result = []
    hangul_decomposed = hgtk.text.decompose(hangul_letter[0])
    hangul_decomposed = \
        hangul_decomposed.replace(hgtk.text.DEFAULT_COMPOSE_CODE, '')
    for i in range(len(hangul_decomposed)):
        hangul = hangul_decomposed[i]
        if i == 0 and hangul in MATCH_H2B_CHO:
            result.append(MATCH_H2B_CHO[hangul])
        if i == 0 and hangul in MATCH_H2B_ALPHABET:
            result.append(MATCH_H2B_ALPHABET[hangul])
        if i == 1 and hangul in MATCH_H2B_JOONG:
            result.append(MATCH_H2B_JOONG[hangul])
        if i == 2 and hangul in MATCH_H2B_JONG:
            result.append(MATCH_H2B_JONG[hangul])
    if result == []:
        result.append([[0,0,0,0,0,0]])
    return result


def text(hangul_sentence):
    result = []

    for hangul_letter in hangul_sentence:
        result.append(letter(hangul_letter))
    return result

txt_braille = text(pt.image_to_string("test_07.jpg",lang = 'kor'))
for i in range (3):
    txt_braille = [item for sublist in txt_braille for item in sublist]

for i in range (0,6):
    print(1-txt_braille[i])
