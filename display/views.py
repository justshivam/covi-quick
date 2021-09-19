from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .predictor import predict
import os

PATH = './media/data'
# Create your views here.
def clear_mediadir():
    media_dir = PATH
    for f in os.listdir(media_dir):
        os.remove(os.path.join(media_dir, f))

def index(request):
    if request.method == 'POST':
        clear_mediadir()
        img = request.FILES['imgfile']
        print(type(img))

        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        print(fs.path(filename))
        res = 'Positive' if predict() == 0 else 'Negative'
        return render(request, 'res.html', {'res': res})
    else:
        return render(request, 'index.html')
