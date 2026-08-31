
class PorterStemmer:

    def __init__(self):
        self.b = ""   # buffer holding the word being stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0    # index used inside various routines

    def cons(self, i):
        """True if b[i] is a consonant."""
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == 'y':
            return True if i == self.k0 else not self.cons(i - 1)
        return True

    def m(self):
        """Measure: number of consonant sequences between k0 and j."""
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i += 1
            i += 1

    def vowelinstem(self):
        return any(not self.cons(i) for i in range(self.k0, self.j + 1))

    def doublec(self, j):
        if j < self.k0 + 1:
            return False
        return self.b[j] == self.b[j - 1] and self.cons(j)

    def cvc(self, i):
        if i < self.k0 + 2 or not self.cons(i) or self.cons(i - 1) or not self.cons(i - 2):
            return False
        ch = self.b[i]
        return ch not in "wxy"

    def ends(self, s):
        length = len(s)
        if s[-1] != self.b[self.k]:
            return False
        if length > (self.k - self.k0 + 1):
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def setto(self, s):
        self.b = self.b[:self.j + 1] + s
        self.k = self.j + len(s)

    def r(self, s):
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k -= 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k -= 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):
                self.setto("ate")
            elif self.ends("bl"):
                self.setto("ble")
            elif self.ends("iz"):
                self.setto("ize")
            elif self.doublec(self.k):
                self.k -= 1
                if self.b[self.k] in "lsz":
                    self.k += 1
            elif self.m() == 1 and self.cvc(self.k):
                self.setto("e")

    def step1c(self):
        if self.ends("y") and self.vowelinstem():
            self.b = self.b[:self.k] + 'i' + self.b[self.k + 1:]

    def step2(self):
        table = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
            ("logi", "log"),
        ]
        for suf, rep in table:
            if self.ends(suf):
                self.r(rep)
                break

    def step3(self):
        table = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suf, rep in table:
            if self.ends(suf):
                self.r(rep)
                break

    def step4(self):
        suffixes = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant",
                    "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous",
                    "ive", "ize"]
        for suf in suffixes:
            if self.ends(suf):
                if self.m() > 1:
                    self.k = self.j
                return
        if self.ends("ion"):
            if self.m() > 1 and self.j >= self.k0 and self.b[self.j] in "st":
                self.k = self.j

    def step5(self):
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k -= 1

    def stem(self, word):
        word = word.lower()
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        if self.k <= self.k0 + 1:
            return self.b
        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k + 1]


_stemmer = PorterStemmer()


def stem_word(word):
    return _stemmer.stem(word)


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        word = line.strip()
        print(stem_word(word))