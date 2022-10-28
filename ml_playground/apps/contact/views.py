import bleach
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.core.mail import send_mail
from ml_playground.apps.contact.models import Contact

from ml_playground import settings

from .forms import ContactForm

def contact(request: HttpRequest) -> HttpResponse:
    if request.method == 'GET':
        form = ContactForm()
    elif request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name = bleach.clean(form.cleaned_data["name"])
            email = bleach.clean(form.cleaned_data["email"])
            message = bleach.clean(form.cleaned_data["message"])
            contact = Contact(name=name, email=email, message=message)
            contact.save()
            send_mail(f"{name} sent an email | {email}", message, settings.DEFAULT_FROM_EMAIL, [settings.DEFAULT_FROM_EMAIL])
            return render(request, "contact.html", {"form": ContactForm(), "success": True})
    else:
        raise NotImplementedError
    return render(request, "contact.html", {"form": form})