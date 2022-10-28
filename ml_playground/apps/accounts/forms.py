from django import forms

class SignupForm(forms.Form):
    first_name = forms.CharField(required=True, widget=forms.TextInput(attrs={'class' : 'form-control', 'id' : 'first_name', 'placeholder' : 'First Name', 'data-sb-validations' : 'required'}))
    last_name = forms.CharField(required=True, widget=forms.TextInput(attrs={'class' : 'form-control', 'id' : 'last_name', 'placeholder' : 'Last Name', 'data-sb-validations' : 'required'}))
    username = forms.CharField(required=True, widget=forms.TextInput(attrs={'class' : 'form-control', 'id' : 'username', 'placeholder' : 'Username', 'data-sb-validations' : 'required'}))
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class' : 'form-control', 'id' : 'email', 'placeholder' : 'Email'}))
    password = forms.CharField(required=True, widget=forms.PasswordInput(attrs={'class' : 'form-control', 'id' : 'password', 'placeholder' : 'Password', 'data-sb-validations' : 'required'}))
    confirm_password = forms.CharField(required=True, widget=forms.PasswordInput(attrs={'class' : 'form-control', 'id' : 'confirm_password', 'placeholder' : 'Confirm Password', 'data-sb-validations' : 'required'}))
