import bleach
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from django.contrib.auth.models import User
from ml_playground.apps.boards.models import Board, BoardVersion, BoardUser
from .forms import BoardForm
from ml_playground.apps.boards.custom_enc_dec import CustomEncDec

@login_required
def index(request):
    boards_ids = list(Board.objects.filter(owner_id = request.user.id).values_list('id', flat=True))
    boards_ids += list(BoardUser.objects.filter(user = request.user.id).values_list('board_id', flat=True))
    boards = Board.objects.filter(pk__in=boards_ids).order_by('-created_at')
    return render(request, "boards/index.html", {'boards': boards})

@login_required
def show(request, id):
    board = Board.objects.get(reference=id)
    board_users = list(BoardUser.objects.filter(board_id=board.id).values_list('user_id', flat=True))
    if request.user.id in board_users:
        return render(request, "boards/show.html", {'board': board, 'can_write': board.can_write(request.user.id)})
    else:
        messages.error(request, "You don't have permission to access this board!")
        return render(request, "boards/index.html")

@login_required
def new(request):
    if request.method == 'GET':
        form = BoardForm()
        return render(request, "boards/new.html", {'form': form})
    elif request.method == 'POST':
        form = BoardForm(request.POST)
        if form.is_valid():
            board = Board(owner=request.user)
            board.set_reference_number()
            custom_enc_dec = CustomEncDec()
            enc_name = custom_enc_dec.encrypt(bleach.clean(form.cleaned_data["name"]), board.reference)
            board.name = enc_name
            board.save()
            board_user = BoardUser(user=request.user, permission='owner', board=board)
            board_user.save()
            version = BoardVersion(name=enc_name, content="", board=board, modified_by=request.user)
            version.save()
            messages.success(request, "Board created successfully.")
            return redirect('boards:edit', id = board.reference)

@login_required
def edit(request, id):
    if request.method == 'GET':
        board = Board.objects.get(reference=id)
        board_users_ids = list(BoardUser.objects.filter(board_id=board.id).values_list('user_id', flat=True))
        if (request.user.id in board_users_ids) and (board.can_write(request.user.id)):
            version = BoardVersion.objects.filter(board_id=board.id).latest('id')
            return render(request, "boards/edit.html", {'board': board, 'version': version})
        else:
            messages.error(request, "You don't have permission to access this page!")
            return render(request, "boards/index.html")
    elif request.method == 'POST':
        custom_enc_dec = CustomEncDec()
        board = Board.objects.get(reference=id)
        enc_content = custom_enc_dec.encrypt(request.POST.get("content"), board.reference)
        previous_name = board.latest_name()
        previous_name = '' if (previous_name is None) else previous_name
        version = BoardVersion(name=previous_name, content=enc_content, board=board, modified_by=request.user)
        version.save()
        messages.success(request, f"Changes on the board saved successfully.")
        return redirect('boards:show', id = board.reference)

@login_required
def rename(request, id):
    board = Board.objects.get(reference=id)
    if request.method == 'GET':
        board_users_ids = list(BoardUser.objects.filter(board_id=board.id).values_list('user_id', flat=True))
        if (request.user.id in board_users_ids) and (board.can_write(request.user.id)):
            version = BoardVersion.objects.filter(board_id=board.id).latest('id')
            form = BoardForm()
            return render(request, "boards/rename.html", {'form': form, 'board': board, 'version': version})
        else:
            messages.error(request, "You don't have permission to access this page!")
            return render(request, "boards/index.html")
    elif request.method == 'POST':
        custom_enc_dec = CustomEncDec()
        enc_name = custom_enc_dec.encrypt(request.POST.get("name"), board.reference)
        previous_content = board.latest_content()
        previous_content = '' if (previous_content is None) else previous_content
        version = BoardVersion(name=enc_name, content=previous_content, board=board, modified_by=request.user)
        version.save()
        messages.success(request, f"Board renamed successfully.")
        return redirect('boards:show', id = board.reference)

@login_required
def share(request, id):
    if request.method == 'GET':
        board = Board.objects.get(reference=id)
        board_users = BoardUser.objects.filter(board_id=board.id)
        board_users_ids = list(BoardUser.objects.filter(board_id=board.id).values_list('user_id', flat=True))
        if (request.user.id in board_users_ids) and (board.can_write(request.user.id)):
            existing_user_ids = BoardUser.objects.filter(board_id=board.id).values_list('user_id', flat=True)
            data_for_options = User.objects.all().exclude(id__in=existing_user_ids).values_list('id', 'first_name', 'last_name')
            return render(request, "boards/share.html", {'board': board, 'data_for_options': data_for_options, 'board_users': board_users})
        else:
            messages.error(request, "You don't have permission to access this page!")
            return render(request, "boards/index.html")
    elif request.method == 'POST':
        board = Board.objects.get(reference=id)
        user_id = bleach.clean(request.POST.get("user_id"))
        user = User.objects.get(pk=user_id)
        permission = bleach.clean(request.POST.get("permission"))
        board_user = BoardUser(board=board, user_id=user_id, permission=permission)
        board_user.save()
        messages.success(request, f"Board shared with {user.get_full_name()} successfully.")
        return redirect('boards:share', id = board.reference)

@login_required
def remove_board_user(request, board_user_id):
    board_user = BoardUser.objects.get(pk=board_user_id)
    if request.user.id == board_user.board.owner_id:
        board_user.delete()
        messages.success(request, "User removed successfully from the board.")
    else:
        messages.error(request, "You don't have rights to remove a user!")
    return redirect('boards:share', id = board_user.board.reference)

@login_required
def versions(request, id):
    board = Board.objects.get(reference=id)
    board_users = list(BoardUser.objects.filter(board_id=board.id).values_list('user_id', flat=True))
    if (request.user.id in board_users) and (board.can_write(request.user.id)):
        versions = board.versions().order_by('-created_at')
        return render(request, "boards/versions.html", {'board': board, 'versions': versions})
    else:
        messages.error(request, "You don't have permission to access this page!")
        return render(request, "boards/index.html")

@login_required
def restore_version(request, version_id):
    version = BoardVersion.objects.get(pk=version_id)
    previous_name = version.board.latest_name()
    previous_name = '' if (previous_name is None) else previous_name
    new_version = BoardVersion(name=previous_name, content=version.content, board=version.board, modified_by=request.user)
    new_version.save()
    messages.success(request, "Board version restored successfully.")
    return redirect('boards:show', id = new_version.board.reference)
