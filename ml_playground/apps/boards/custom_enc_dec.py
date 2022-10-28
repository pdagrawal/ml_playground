from mirage.crypto import Crypto

class CustomEncDec():
    def __init__(self):
        self.characters = [" ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".",
            "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
            "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d",
            "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
            "w", "x", "y", "z"]

    def encrypt(self, plain_text, key):
        c = Crypto()
        cipher_text = ''
        for char in list(plain_text):
            if char in self.characters:
                cipher_text += self.characters[(self.characters.index(char) + self.shift_by(key)) % len(self.characters)]
            else:
                cipher_text += char
        return c.encrypt(cipher_text)

    def decrypt(self, cipher_text, key):
        c = Crypto()
        cipher_text = c.decrypt(cipher_text)
        plain_text = ''
        for char in list(cipher_text):
            if char in self.characters:
                plain_text += self.characters[(self.characters.index(char) - self.shift_by(key) + len(self.characters)) % len(self.characters)]
            else:
                plain_text += char
        return plain_text

    def shift_by(self, key):
        return sum(ord(ch) for ch in key) % 26
