from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(required=True, widget=forms.TextInput(attrs={'class' : 'form-control', 'id' : 'name', 'placeholder' : 'Name', 'data-sb-validations' : 'required'}))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class' : 'form-control', 'id' : 'email', 'placeholder' : 'Email'}))
    message = forms.CharField(required=True, widget=forms.Textarea(attrs={'class' : 'form-control', 'id' : 'message', 'placeholder' : 'Message', 'rows' : 5, 'style' : 'height: 10rem'}))
