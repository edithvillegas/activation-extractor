import requests
from tqdm.auto import tqdm

# from 
def download_file(url, filename):
  """
  to download prottrans models from dropbox
  from https://github.com/agemagician/ProtTrans/
  """
  response = requests.get(url, stream=True)
  with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                    total=int(response.headers.get('content-length', 0)),
                    desc=filename) as fout:
      for chunk in response.iter_content(chunk_size=4096):
          fout.write(chunk)