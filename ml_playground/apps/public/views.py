from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
from django.contrib import messages


def index(request: HttpRequest) -> HttpResponse:
    return render(request, "index.html")


def about(request: HttpRequest) -> HttpResponse:
    return render(request, "about.html")

def upload_csv(request: HttpRequest) -> HttpResponse:
    # return redirect('public:index')
    field_names = []
    if "GET" == request.method:
        return render(request, "public/upload_csv.html", field_names)
    # if not GET, then proceed
    try:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'File is not CSV type')
            return render(request, "public/index.html")
        #if file is too large, return
        if csv_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)." % (csv_file.size/(1000*1000),))
            return render(request, "public/index.html")

        file_data = csv_file.read().decode("utf-8")

        lines = file_data.split("\n")
        field_names = lines[0].split(',')
        #loop over the lines and save them in db. If error , store as string and then display
        # for line in lines:
        #     fields = line.split(",")
        #     data_dict = {}
        #     data_dict["name"] = fields[0]
        #     data_dict["start_date_time"] = fields[1]
        #     data_dict["end_date_time"] = fields[2]
        #     data_dict["notes"] = fields[3]
        #     try:
        #         form = EventsForm(data_dict)
        #         if form.is_valid():
        #             form.save()
        #         else:
        #             logging.getLogger("error_logger").error(form.errors.as_json())
        #     except Exception as e:
        #         logging.getLogger("error_logger").error(repr(e))
        #         pass

    except Exception as e:
        # logging.getLogger("error_logger").error("Unable to upload file. "+repr(e))
        messages.error(request, "Unable to upload file. "+repr(e))

    return render(request, "field_selection.html", {'field_names': field_names})
