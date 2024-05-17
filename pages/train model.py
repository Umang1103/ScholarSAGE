import tempfile
import streamlit as st
import datetime
import shutil
from initializations import *
from utils import PrepareDocsAndEmbeddings

st.set_page_config(page_icon="🎓")
st.header("ScholarSAGE")
st.subheader("Make your assistant learn new data")

uploaded_files = st.file_uploader('Upload your .pdf file', type="pdf", accept_multiple_files=True)

if st.button("Process"):
    if len(uploaded_files) == 0:
        st.warning("Please upload the file!")

    elif uploaded_files is not None:
        with st.status("Processing documents"):
            # create temporary directory to store pdfs
            temp_dir = tempfile.mkdtemp()
            print(f"{datetime.datetime.now()} Temporarily created directory {temp_dir}.")

            # Write pdfs in the directory
            st.write("Uploading pdfs...")
            for file in uploaded_files:
                temp_pdf_path = os.path.join(temp_dir, file.name)
                with open(temp_pdf_path, mode='wb') as w:
                    w.write(file.getvalue())
                print(f"{datetime.datetime.now()} Temporarily saving {temp_pdf_path}.")

            # prepare and upload the embeddings to the vector database
            st.write("Preparing vector embeddings...")
            vector_db = PrepareDocsAndEmbeddings(path_to_directory=temp_dir).get_embeddings(prepare_embeddings=True)

            # delete temporary directory
            shutil.rmtree(temp_dir)
            st.write("Model ready to run!")
            print(f"{datetime.datetime.now()} Temporary directory deleted.")
