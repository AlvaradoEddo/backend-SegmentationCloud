from time import sleep
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from typing import List

app = FastAPI()


file = None;

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

@app.post("/uploadfile/")
def create_upload_file(files: UploadFile):
    if files and files.content_type == "image/*":
        sleep(100)
        process_image(files.file)
        print("prueba de return" )
        return {"filename": files.filename}

    raise HTTPException(status_code=404, detail="File is not a tiff image")

def process_image(image: bytes = File(...)):
    #print(image.read())
    print("prueba")
    #return {"image_size": len(image)}

@app.post("/execute")
async def execute():
    return {"status": "ok"}


@app.get("/finalImage")
async def get_final_image():
    return {"image": "image"}

@app.get("/result/")
async def read_result():
    return {"result": "result"}