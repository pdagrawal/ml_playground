import uuid
from django.db import models
from django.contrib.auth.models import User

from ml_playground.apps.boards.custom_enc_dec import CustomEncDec

class Board(models.Model):
    name = models.CharField(max_length=200)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    reference = models.CharField(max_length=200, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def owner_name(self):
        return self.owner.get_full_name()

    def decrypted_name(self):
        return self.decrypt(self.latest_name())

    def latest_name(self):
        return BoardVersion.objects.filter(board_id=self.id).latest('id').name

    def decrypt(self, body):
        custom_enc_dec = CustomEncDec()
        return custom_enc_dec.decrypt(body, self.reference)

    def decrypted_content(self):
        version = BoardVersion.objects.filter(board_id=self.id).latest('id')
        if version is not None:
            return version.decrypted_content()
        else:
            return ''

    def latest_content(self):
        return BoardVersion.objects.filter(board_id=self.id).latest('id').content

    def set_reference_number(self):
        reference_number = uuid.uuid4().hex
        if Board.objects.filter(reference=reference_number).exists():
            self.set_reference_number()
        else:
            self.reference = reference_number
            return self

    def versions(self):
        return BoardVersion.objects.filter(board_id=self.id)

    def can_write(self, user_id):
        return BoardUser.objects.get(board_id=self.id, user_id=user_id).permission != 'read'

class BoardVersion(models.Model):
    name = models.CharField(max_length=300, default="")
    content = models.TextField()
    board = models.ForeignKey(Board, on_delete=models.CASCADE)
    modified_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.board.name

    def modified_by_name(self):
        return self.modified_by.get_full_name()

    def decrypted_name(self):
        custom_enc_dec = CustomEncDec()
        return custom_enc_dec.decrypt(self.name, self.board.reference)

    def decrypted_content(self):
        custom_enc_dec = CustomEncDec()
        return custom_enc_dec.decrypt(self.content, self.board.reference)

class BoardUser(models.Model):
    board = models.ForeignKey(Board, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    permission = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.board.name}:{self.user.get_full_name()}"