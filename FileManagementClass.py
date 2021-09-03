from urllib.request import urlretrieve, urlcleanup
import zipfile
import os
class FileManagement:
    def downloadFile(url, path):
        file = path+"/file.zip"
        urlretrieve(url, file)
        urlcleanup()
        return file

    def extractFile(file, path):
        password = None
        archivo_zip = zipfile.ZipFile(file, "r")

        try:
            name = archivo_zip.namelist()
            archivo_zip.extractall(pwd=password, path=path)
            print(name[0])
            directoryname = name[0]
        except:
            pass
        archivo_zip.close()
        os.remove(file)
        return directoryname
