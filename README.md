# Document_Scanner #
This project is made in python3 by using a very strong computer vision library opencv. It can scan documents from images and videos and convert them into pdf.

## Dependencies: ##
* python 3.6.0
* opencv 4.4.0
* PILLOW 7.2.0

## How to use ##
```python
if __name__ == "__main__":
    doc = DocumentScanner('vid.mp4', IMG_FLAG=False, VID_FLAG=True, SAVE_PDF=True)
    doc.execute()
```
    
### parameters ### 
* `path`: path to the file(image or video)
* `IMG_FLAG` : Flag if image is used or not. By default `False`. If `path` is image then `True`
* `VID_FLAG` : Flag if video is used or not. By default `False`. If `path` is video then `True`
* `SAVE_PDF` : Flag if pdf is to be saved or not. By default `False`
* `path_to_save` : directory where the pdf is to be saved. By default `None`. But if `SAVE_PDF=True` then it has to be non-empty.

### How to close ###
* Just press `q` on keyboard.
