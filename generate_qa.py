from glob import glob
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from zindi_llm.qa_generation import make_document_prediction

files = sorted(glob("data/zindi_data/rel18/*.docx"))
file = files[1]

name = os.path.basename(file).replace(".docx", "")
make_document_prediction(
    file, file_name=name, batch_size=8, n_sentence=70
)