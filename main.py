from time import sleep
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from model import get_model,get_img_proccessed,get_base64_img_from_mask
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file = None
ResNet_model2=get_model()
@app.get("/")
async def main():
    content = """
<body>
<form action="/uploadfile/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

@app.post("/files/")
async def create_upload_file(file: bytes = File(...)):
    return {"file_size": len(file)}
import re
@app.post("/uploadfile/")
def create_upload_file(files: UploadFile):
    print(files.content_type)
    regex_all_img = re.compile(r"image/.*")
    if files and regex_all_img.match(files.content_type):
        print("prueba de if", type(files))

        base64_img=process_image(files.file)
        print("prueba de return" ,base64_img)
        #return as json
        return {"image": base64_img}

    raise HTTPException(status_code=404, detail="File is not a image")


def process_image(image: bytes = File(...)):

    output=get_img_proccessed(image,ResNet_model2)
    mask = output[0][1]
    print("prueba de mask", mask)
    img_str = get_base64_img_from_mask(mask)

    return img_str


@app.post("/execute")
async def execute():
    return {"status": "ok"}


@app.get("/finalImage")
async def get_final_image():
    return {"image": "image"}

@app.get("/result/")
async def read_result():
    return {"result": "result"}