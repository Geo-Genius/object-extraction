import tempfile

file = tempfile.TemporaryFile(suffix=".tiff")
print(file)
print(file.name)