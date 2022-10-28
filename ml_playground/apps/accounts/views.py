import bleach
from django.contrib.auth import login
from django.core.mail import EmailMessage, send_mail
from django.shortcuts import redirect, render
from django.http import HttpRequest, HttpResponse
from django.views.generic.base import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_text

from . tokens import generate_token
from . forms import SignupForm
from ml_playground import settings


class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = "accounts/profile.html"

def signup(request: HttpRequest) -> HttpResponse:
    if request.method == 'GET':
        form = SignupForm()
    elif request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            username = bleach.clean(form.cleaned_data["username"])
            first_name = bleach.clean(form.cleaned_data["first_name"])
            last_name = bleach.clean(form.cleaned_data["last_name"])
            email = bleach.clean(form.cleaned_data["email"])
            password = bleach.clean(form.cleaned_data["password"])
            confirm_password = bleach.clean(form.cleaned_data["confirm_password"])

            if User.objects.filter(username = username):
                messages.error(request, 'Username already taken! Please try another username')

            if User.objects.filter(email = email):
                messages.error(request, 'Email already registered!')

            if len(username) < 3:
                messages.error(request, 'Username must be at least 3 character long!')

            if password != confirm_password:
                messages.error(request, "Passwords don't match!")

            if not username.isalnum():
                messages.error(request, "Username must be alphanumeric!")

            if len(list(messages.get_messages(request))) > 0:
                return render(request, "accounts/signup.html", {"form": form})

            user = User.objects.create_user(username, email, password)
            user.first_name = first_name
            user.last_name = last_name
            user.is_active = False
            user.save()
            messages.success(request, "Your account has been successfully created. We have sent you a confirmation email. Please check to activate your account.")
            current_site = get_current_site(request)
            email_subject = "Welcome to CryptoBoard! Confirm Your Email"
            email_content = render_to_string('accounts/signup_email.html', {
                'first_name': first_name,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': generate_token.make_token(user)
            })
            email = EmailMessage(
                email_subject,
                email_content,
                settings.DEFAULT_FROM_EMAIL,
                [email]
            )
            email.fail_silently = True
            email.send()

            return redirect('accounts:login')
    else:
        raise NotImplementedError
    return render(request, "accounts/signup.html", {"form": form})

def activate(request, uidb64, token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and generate_token.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)
        return redirect('public:index')
    else:
        return render(request, 'accounts/activation_failed.html')