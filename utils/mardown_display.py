# source: https://stackoverflow.com/questions/75610911/how-to-use-mermaid-diagram-in-jupyter-notebook-with-mermaid-ink-through-proxy
# enable mermaid rendering in cells
# use mm(""" graph goes here; """)
import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt

def mm(graph):
  graphbytes = graph.encode("ascii")
  base64_bytes = base64.b64encode(graphbytes)
  base64_string = base64_bytes.decode("ascii")
  display(
    Image(
      url="https://mermaid.ink/img/"
      + base64_string
    )
  )