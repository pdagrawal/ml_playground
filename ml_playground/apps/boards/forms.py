from django import forms

class BoardForm(forms.Form):
    name = forms.CharField(required=True, widget=forms.TextInput(attrs={'class' : 'form-control', 'id' : 'name', 'placeholder' : 'Name', 'data-sb-validations' : 'required'}))
