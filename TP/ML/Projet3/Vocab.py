import torch

class Vocab:
    def __init__(self, **kwargs):
        self.dico_voca = {}
        self.word_array = []
        if "emb_filename" in kwargs:
            with open(kwargs["emb_filename"], 'r', encoding='utf-8') as fi:
                ligne = fi.readline()
                ligne = ligne.strip()
                (self.vocab_size, self.emb_dim) = map(int, ligne.split(" "))
                self.matrice = torch.zeros((self.vocab_size, self.emb_dim))
                indice_mot = 0

                ligne = fi.readline()
                ligne = ligne.strip()
                while ligne != '':
                    splitted_ligne = ligne.split()
                    self.dico_voca[splitted_ligne[0]] = indice_mot
                    self.word_array.append(splitted_ligne[0])
                    for i in range(1, len(splitted_ligne)):
                        self.matrice[indice_mot, i - 1] = float(splitted_ligne[i])
                    indice_mot += 1
                    ligne = fi.readline()
                    ligne = ligne.strip()
        else:
            fichier_corpus = kwargs["corpus_filename"]
            self.emb_dim = kwargs["emb_dim"]
            nb_tokens = 0
            with open(fichier_corpus, 'r', encoding='utf-8') as fi:
                for line in fi:
                    line = line.rstrip()
                    tokens = line.split(" ")
                    for token in tokens:
                        if token not in self.dico_voca:
                            self.word_array.append(token)
                            self.dico_voca[token] = nb_tokens
                            nb_tokens += 1
            self.vocab_size = nb_tokens
            print("vocab size =", self.vocab_size, "emb_dim =", self.emb_dim)
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim))

    def get_word_index(self, mot):
        if not mot in self.dico_voca:
            return None
        return self.dico_voca[mot]

    def get_word_index2(self, mot):
        if not mot in self.dico_voca:
            return self.dico_voca['<unk>']
        return self.dico_voca[mot]

    def get_emb(self, mot):
        if not mot in self.dico_voca:
            return None
        return self.matrice[self.dico_voca[mot]]

    def get_emb_torch(self, indice_mot):
        return self.matrice[indice_mot]

    def get_one_hot(self, mot):
        vect = torch.zeros(len(self.dico_voca))
        vect[self.dico_voca[mot]] = 1
        return vect

    def get_word(self, index):
        if index < len(self.word_array):
            return self.word_array[index]
        else:
            return None

    def get_vocab_dict(self):
        return self.dico_voca